# LVD MODIFICATION START
# Note: this file is inspired by the file chromadb/segment/impl/vector/local_persistent_hnsw.py

import os
import shutil
from overrides import override
import pickle
from typing import Dict, List, Optional, Sequence, Set, cast
from chromadb.config import System
from chromadb.segment.impl.vector.batch import Batch
from chromadb.segment.impl.vector.lmi_params import PersistentLMIParams
from chromadb.segment.impl.vector.local_lmi import (
    DEFAULT_CAPACITY,
    LocalLMISegment,
)
from chromadb.segment.impl.vector.brute_force_index import BruteForceIndex
from chromadb.telemetry.opentelemetry import (
    OpenTelemetryClient,
    OpenTelemetryGranularity,
    trace_method,
)
from chromadb.types import (
    EmbeddingRecord,
    Metadata,
    Operation,
    Segment,
    SeqId,
    Vector,
    VectorEmbeddingRecord,
    VectorQuery,
    VectorQueryResult,
)
import logging
import numpy as np
from chromadb.li_index.search.lmi import LMI

from chromadb.utils.read_write_lock import ReadRWLock, WriteRWLock


logger = logging.getLogger(__name__)


class PersistentData:
    """Stores the data and metadata needed for a PersistentLocalLMISegment"""

    dimensionality: Optional[int]
    total_elements_added: int
    max_seq_id: SeqId

    id_to_label: Dict[str, int]
    label_to_id: Dict[int, str]
    id_to_seq_id: Dict[str, SeqId]

    def __init__(
        self,
        dimensionality: Optional[int],
        total_elements_added: int,
        max_seq_id: int,
        id_to_label: Dict[str, int],
        label_to_id: Dict[int, str],
        id_to_seq_id: Dict[str, SeqId],
    ):
        self.dimensionality = dimensionality
        self.total_elements_added = total_elements_added
        self.max_seq_id = max_seq_id
        self.id_to_label = id_to_label
        self.label_to_id = label_to_id
        self.id_to_seq_id = id_to_seq_id

    @staticmethod
    def load_from_file(filename: str) -> "PersistentData":
        """Load persistent data from a file"""
        with open(filename, "rb") as f:
            ret = cast(PersistentData, pickle.load(f))
            return ret


class PersistentLocalLMISegment(LocalLMISegment):
    METADATA_FILE: str = "index_metadata.pickle"
    INDEX_FILE: str = "index.pickle"
    # How many records to add to index at once, we do this because crossing the python/c++ boundary is expensive (for add())
    # When records are not added to the c++ index, they are buffered in memory and served
    # via brute force search.
    _batch_size: int
    _brute_force_index: Optional[BruteForceIndex]
    _index_initialized: bool = False
    _curr_batch: Batch
    # How many records to add to index before syncing to disk
    _sync_threshold: int
    _persist_data: PersistentData
    _persist_directory: str
    _allow_reset: bool

    _opentelemtry_client: OpenTelemetryClient

    def __init__(self, system: System, segment: Segment):
        super().__init__(system, segment)

        self._opentelemtry_client = system.require(OpenTelemetryClient)

        self._params = PersistentLMIParams(segment["metadata"] or {})
        self._batch_size = self._params.batch_size
        self._sync_threshold = self._params.sync_threshold
        self._allow_reset = system.settings.allow_reset
        self._persist_directory = system.settings.require("persist_directory")
        self._curr_batch = Batch()
        self._brute_force_index = None
        if not os.path.exists(self._get_storage_folder()):
            os.makedirs(self._get_storage_folder(), exist_ok=True)
        # Load persist data if it exists already, otherwise create it
        if self._index_exists():
            self._persist_data = PersistentData.load_from_file(
                self._get_metadata_file()
            )
            self._dimensionality = self._persist_data.dimensionality
            self._total_elements_added = self._persist_data.total_elements_added
            self._max_seq_id = self._persist_data.max_seq_id
            self._id_to_label = self._persist_data.id_to_label
            self._label_to_id = self._persist_data.label_to_id
            self._id_to_seq_id = self._persist_data.id_to_seq_id
            # If the index was written to, we need to re-initialize it
            if len(self._id_to_label) > 0:
                self._dimensionality = cast(int, self._dimensionality)
                self._init_index(self._dimensionality)
        else:
            self._persist_data = PersistentData(
                self._dimensionality,
                self._total_elements_added,
                self._max_seq_id,
                self._id_to_label,
                self._label_to_id,
                self._id_to_seq_id,
            )

    @staticmethod
    @override
    def propagate_collection_metadata(metadata: Metadata) -> Optional[Metadata]:
        # Extract relevant metadata
        segment_metadata = PersistentLMIParams.extract(metadata)
        return segment_metadata

    def _index_exists(self) -> bool:
        """Check if the index exists via the metadata file"""
        return os.path.exists(self._get_metadata_file())

    def _get_metadata_file(self) -> str:
        """Get the metadata file path"""
        return os.path.join(self._get_storage_folder(), self.METADATA_FILE)

    def _get_storage_folder(self) -> str:
        """Get the storage folder path"""
        folder = os.path.join(self._persist_directory, str(self._id))
        return folder

    @override
    def _init_index(self, dimensionality: int) -> None:
        index = LMI()
        self._brute_force_index = BruteForceIndex(
            size=self._batch_size,
            dimensionality=dimensionality,
            space=self._params.space,
        )

        # Check if index exists and load it if it does
        if self._index_exists():
            # index = self.load_lmi_from_pickle(os.path.join(self._get_storage_folder(), self.INDEX_FILE))
            index.load_index(
                self._get_storage_folder(),
                is_persistent_index=True,
            )
        else:
            index.init_index(
                max_elements=DEFAULT_CAPACITY,
                clustering_algorithms=self._params.clustering_algorithms,
                epochs=self._params.epochs,
                learning_rate=self._params.lrs,
                model_types=self._params.model_types,
                n_categories=self._params.n_categories,
                kmeans=self._params.kmeans,
                is_persistent_index=True,
                persistence_location=self._get_storage_folder(),
            )

        index.set_num_threads(self._params.num_threads)

        self._index = index
        self._dimensionality = dimensionality
        self._index_initialized = True

    def _persist(self, persist_index = False) -> None:
        """Persist the index and data to disk"""
        index = cast(LMI, self._index)

        # Persist the index
        # file_path = os.path.join(self._get_storage_folder(), self.INDEX_FILE)
        #
        # with open(file_path, 'wb') as f:
        #     pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

        index.persist_dirty()

        # Persist the metadata
        self._persist_data.dimensionality = self._dimensionality
        self._persist_data.total_elements_added = self._total_elements_added
        self._persist_data.max_seq_id = self._max_seq_id

        # TODO: This should really be stored in sqlite, the index itself, or a better
        # storage format
        self._persist_data.id_to_label = self._id_to_label
        self._persist_data.label_to_id = self._label_to_id
        self._persist_data.id_to_seq_id = self._id_to_seq_id

        with open(self._get_metadata_file(), "wb") as metadata_file:
            pickle.dump(self._persist_data, metadata_file, pickle.HIGHEST_PROTOCOL)

    @override
    def _apply_batch(self, batch: Batch) -> None:
        super()._apply_batch(batch)
        if (
            self._total_elements_added - self._persist_data.total_elements_added
            >= self._sync_threshold
        ):
            self._persist()

    @override
    def build_index(self) -> Dict[str, List[int]]:
        id_to_bucket = super().build_index()
        self._persist(True)
        return id_to_bucket

    @override
    def _write_records(self, records: Sequence[EmbeddingRecord]) -> None:
        """Add a batch of embeddings to the index"""
        if not self._running:
            raise RuntimeError("Cannot add embeddings to stopped component")
        with WriteRWLock(self._lock):
            for record in records:
                if record["embedding"] is not None:
                    self._ensure_index(len(records), len(record["embedding"]))
                if not self._index_initialized:
                    # If the index is not initialized here, it means that we have
                    # not yet added any records to the index. So we can just
                    # ignore the record since it was a delete.
                    continue
                self._brute_force_index = cast(BruteForceIndex, self._brute_force_index)

                self._max_seq_id = max(self._max_seq_id, record["seq_id"])
                id = record["id"]
                op = record["operation"]
                exists_in_index = self._id_to_label.get(
                    id, None
                ) is not None or self._brute_force_index.has_id(id)
                exists_in_bf_index = self._brute_force_index.has_id(id)

                if op == Operation.DELETE:
                    if exists_in_index:
                        self._curr_batch.apply(record)
                        if exists_in_bf_index:
                            self._brute_force_index.delete([record])
                    else:
                        logger.warning(f"Delete of nonexisting embedding ID: {id}")

                elif op == Operation.UPDATE:
                    if record["embedding"] is not None:
                        if exists_in_index:
                            self._curr_batch.apply(record)
                            self._brute_force_index.upsert([record])
                        else:
                            logger.warning(
                                f"Update of nonexisting embedding ID: {record['id']}"
                            )
                elif op == Operation.ADD:
                    if record["embedding"] is not None:
                        if not exists_in_index:
                            self._curr_batch.apply(record, not exists_in_index)
                            self._brute_force_index.upsert([record])
                        else:
                            logger.warning(f"Add of existing embedding ID: {id}")
                elif op == Operation.UPSERT:
                    if record["embedding"] is not None:
                        self._curr_batch.apply(record, exists_in_index)
                        self._brute_force_index.upsert([record])
                if len(self._curr_batch) >= self._batch_size:
                    self._apply_batch(self._curr_batch)
                    self._curr_batch = Batch()
                    self._brute_force_index.clear()

    @override
    def count(self) -> int:
        return (
            len(self._id_to_label)
            + self._curr_batch.add_count
            - self._curr_batch.delete_count
        )

    @override
    def get_vectors(
        self, ids: Optional[Sequence[str]] = None
    ) -> Sequence[VectorEmbeddingRecord]:
        """Get the embeddings from the HNSW index and layered brute force
        batch index."""

        ids_lmi: Set[str] = set()
        ids_bf: Set[str] = set()

        if self._index is not None:
            ids_lmi = set(self._id_to_label.keys())
        if self._brute_force_index is not None:
            ids_bf = set(self._curr_batch.get_written_ids())

        target_ids = ids or list(ids_lmi.union(ids_bf))
        self._brute_force_index = cast(BruteForceIndex, self._brute_force_index)
        lmi_labels = []

        results: List[Optional[VectorEmbeddingRecord]] = []
        id_to_index: Dict[str, int] = {}
        for i, id in enumerate(target_ids):
            if id in ids_bf:
                results.append(self._brute_force_index.get_vectors([id])[0])
            elif id in ids_lmi and id not in self._curr_batch._deleted_ids:
                lmi_labels.append(self._id_to_label[id])
                # Placeholder for lmi results to be filled in down below so we
                # can batch the lmi get() call
                results.append(None)
            id_to_index[id] = i

        if len(lmi_labels) > 0 and self._index is not None:
            vectors = cast(Sequence[Vector], self._index.get_items(lmi_labels))

            for label, vector in zip(lmi_labels, vectors):
                id = self._label_to_id[label]
                seq_id = self._id_to_seq_id[id]
                results[id_to_index[id]] = VectorEmbeddingRecord(
                    id=id, seq_id=seq_id, embedding=vector
                )

        return results  # type: ignore ## Python can't cast List with Optional to List with VectorEmbeddingRecord

    @override
    def query_vectors(
        self, query: VectorQuery
    ) -> (Sequence[Sequence[VectorQueryResult]], List[List[int]], bool, float, float):
        if self._index is None:
            return [[] for _ in range(len(query["vectors"]))]

        k = query["k"]
        if k > self.count():
            logger.warning(
                f"Number of requested results {k} is greater than number of elements in index {self.count()}, updating n_results = {self.count()}"
            )
            k = self.count()

        with ReadRWLock(self._lock):
            lmi_results = super().query_vectors(query)

            return lmi_results

    @override
    def reset_state(self) -> None:
        if self._allow_reset:
            data_path = self._get_storage_folder()
            if os.path.exists(data_path):
                self.close_persistent_index()
                shutil.rmtree(data_path, ignore_errors=True)

    @override
    def delete(self) -> None:
        data_path = self._get_storage_folder()
        if os.path.exists(data_path):
            self.close_persistent_index()
            shutil.rmtree(data_path, ignore_errors=False)

    def open_persistent_index(self) -> None:
        """Open the persistent index"""
        if self._index is not None:
            self._index.open_file_handles()

    def close_persistent_index(self) -> None:
        """Close the persistent index"""
        if self._index is not None:
            self._index.close_file_handles()
# LVD MODIFICATION END