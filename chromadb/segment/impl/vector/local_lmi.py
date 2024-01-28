import numpy as np
from overrides import override
from typing import Optional, Sequence, Dict, Set, List, cast
from uuid import UUID
from chromadb.segment import VectorReader
from chromadb.ingest import Consumer
from chromadb.config import System, Settings
from chromadb.segment.impl.vector.batch import Batch
from chromadb.segment.impl.vector.lmi_params import LMIParams
from chromadb.telemetry.opentelemetry import (
    OpenTelemetryClient,
    OpenTelemetryGranularity,
    trace_method,
)
from chromadb.types import (
    EmbeddingRecord,
    VectorEmbeddingRecord,
    VectorQuery,
    VectorQueryResult,
    SeqId,
    Segment,
    Metadata,
    Operation,
    Vector,
)
from chromadb.errors import InvalidDimensionException
from chromadb.utils.read_write_lock import ReadWriteLock, ReadRWLock, WriteRWLock
from chromadb.li_index.search.lmi import LMI
import logging

from chromadb.li_index.search.attribtue_filtering.default_filtering import map_range

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 1000


class LocalLMISegment(VectorReader):
    _id: UUID
    _consumer: Consumer
    _topic: Optional[str]
    _subscription: UUID
    _settings: Settings
    _params: LMIParams

    _index: Optional[LMI]
    _dimensionality: Optional[int]
    _total_elements_added: int
    _max_seq_id: SeqId

    _lock: ReadWriteLock

    _id_to_label: Dict[str, int]
    _label_to_id: Dict[int, str]
    _id_to_seq_id: Dict[str, SeqId]

    _opentelemtry_client: OpenTelemetryClient

    def __init__(self, system: System, segment: Segment):
        self._consumer = system.instance(Consumer)
        self._id = segment["id"]
        self._topic = segment["topic"]
        self._settings = system.settings
        self._params = LMIParams(segment["metadata"] or {})

        self._index = None
        self._dimensionality = None
        self._total_elements_added = 0
        self._max_seq_id = self._consumer.min_seqid()

        self._id_to_seq_id = {'-1': -1}
        self._id_to_label = {'-1': 0}
        self._label_to_id = { 0: '-1'}

        self._lock = ReadWriteLock()
        self._opentelemtry_client = system.require(OpenTelemetryClient)
        super().__init__(system, segment)

    @staticmethod
    @override
    def propagate_collection_metadata(metadata: Metadata) -> Optional[Metadata]:
        # Extract relevant metadata
        segment_metadata = LMIParams.extract(metadata)
        return segment_metadata

    @override
    def start(self) -> None:
        super().start()
        if self._topic:
            seq_id = self.max_seqid()
            self._subscription = self._consumer.subscribe(
                self._topic, self._write_records, start=seq_id
            )

    @override
    def stop(self) -> None:
        super().stop()
        if self._subscription:
            self._consumer.unsubscribe(self._subscription)

    @override
    def get_vectors(
        self, ids: Optional[Sequence[str]] = None
    ) -> Sequence[VectorEmbeddingRecord]:
        if ids is None:
            labels = list(self._label_to_id.keys())
        else:
            labels = []
            for id in ids:
                if id in self._id_to_label:
                    labels.append(self._id_to_label[id])

        results = []
        if self._index is not None:
            vectors = cast(Sequence[Vector], self._index.get_items(labels))

            for label, vector in zip(labels, vectors):
                id = self._label_to_id[label]
                seq_id = self._id_to_seq_id[id]
                results.append(
                    VectorEmbeddingRecord(id=id, seq_id=seq_id, embedding=vector)
                )

        return results

    @override
    def query_vectors(
        self, query: VectorQuery
    ) -> (Sequence[Sequence[VectorQueryResult]], List[List[int]], bool, float, float):
        if self._index is None:
            return [[] for _ in range(len(query["vectors"]))]

        k = query["k"]
        n_buckets = query["n_buckets"]
        bruteforce_threshold = query["bruteforce_threshold"]
        constraint_weight = query["constraint_weight"]
        search_until_bucket_not_empty = query["search_until_bucket_not_empty"]

        size = len(self._id_to_label)

        if k > size:
            logger.warning(
                f"Number of requested results {k} is greater than number of elements in index {size}, updating n_results = {size}"
            )
            k = size

        ids = query["allowed_ids"]

        # CONSTRAINT MODIFICATION START
        # TODO: this might not work with updates and deletes
        use_bruteforce = False
        filter_restrictiveness = 1.0
        if ids is not None:
            filter_restrictiveness = len(ids) / self._total_elements_added
            if filter_restrictiveness < bruteforce_threshold:
                use_bruteforce = True
            elif constraint_weight < 0.0:
                constraint_weight = map_range(1 - filter_restrictiveness, (0.0, 1.0), (0.25, 0.75))
        # CONSTRAINT MODIFICATION END

        if ids is not None:
            filter_ids = [self._id_to_label[id] for id in ids if id in self._id_to_label]

        query_vectors = query["vectors"]

        with ReadRWLock(self._lock):
            result_labels, distances, bucket_order = self._index.knn_query(
                query_vectors,
                k=k,
                n_buckets=n_buckets,
                bruteforce_threshold=bruteforce_threshold,
                constraint_weight=constraint_weight,
                filter=filter_ids if ids is not None else None,
                filter_restrictiveness=filter_restrictiveness,
                use_bruteforce=use_bruteforce,
                search_until_bucket_not_empty=search_until_bucket_not_empty
            )
            bucket_order = bucket_order.tolist()

            # TODO: these casts are not correct, hnswlib returns np
            # distances = cast(List[List[float]], distances)
            # result_labels = cast(List[List[int]], result_labels)

            all_results: List[List[VectorQueryResult]] = []
            for result_i in range(len(result_labels)):
                results: List[VectorQueryResult] = []
                for label, distance in zip(
                    result_labels[result_i], distances[result_i]
                ):
                    id = self._label_to_id[label]
                    seq_id = self._id_to_seq_id[id]
                    if query["include_embeddings"]:
                        # The embeddings are internally represented as pandas data frame
                        # In order to work with FastAPI they need to be converted to list, so they can be used in json
                        embedding = self._index.get_items([label])[0].tolist()
                    else:
                        embedding = None
                    if distance.item() == float("inf"):
                        # FastAPI does not support inf values, since they cannot be serialized to json
                        distance = np.float32(100_000)
                    results.append(
                        VectorQueryResult(
                            id=id,
                            seq_id=seq_id,
                            distance=distance.item(),
                            embedding=embedding,
                        )
                    )
                all_results.append(results)

            return all_results, bucket_order, use_bruteforce, constraint_weight, filter_restrictiveness

    @override
    def max_seqid(self) -> SeqId:
        return self._max_seq_id

    @override
    def count(self) -> int:
        return len(self._id_to_label)

    @override
    def build_index(self) -> Dict[str, List[int]]:
        label_position_to_bucket = self._index.build_index()
        id_to_bucket = {}
        for label, bucket in enumerate(label_position_to_bucket):
            id = self._label_to_id[label + 1]
            id_to_bucket[id] = bucket.tolist()

        return id_to_bucket

    def _init_index(self, dimensionality: int) -> None:
        index = LMI()  # possible options are l2, cosine or ip
        index.init_index(
            max_elements=DEFAULT_CAPACITY,
            clustering_algorithms=self._params.clustering_algorithms,
            epochs=self._params.epochs,
            learning_rate=self._params.lrs,
            model_types=self._params.model_types,
            n_categories=self._params.n_categories,
            kmeans=self._params.kmeans,
        )
        index.set_num_threads(self._params.num_threads)

        self._index = index
        self._dimensionality = dimensionality

    def _ensure_index(self, n: int, dim: int) -> None:
        """Create or resize the index as necessary to accomodate N new records"""
        if not self._index:
            self._dimensionality = dim
            self._init_index(dim)
        else:
            if dim != self._dimensionality:
                raise InvalidDimensionException(
                    f"Dimensionality of ({dim}) does not match index"
                    + f"dimensionality ({self._dimensionality})"
                )

        index = cast(LMI, self._index)
        # The resizing does not have currently effect on the LMI
        if (self._total_elements_added + n) > index.get_max_elements():
            new_size = int(
                (self._total_elements_added + n) * self._params.resize_factor
            )
            index.resize_index(max(new_size, DEFAULT_CAPACITY))

    def _apply_batch(self, batch: Batch) -> None:
        """Apply a batch of changes, as atomically as possible."""
        deleted_ids = batch.get_deleted_ids()
        written_ids = batch.get_written_ids()
        vectors_to_write = batch.get_written_vectors(written_ids)
        labels_to_write = [0] * len(vectors_to_write)

        # Deletion Not Handled By LMI, ignore it
        if len(deleted_ids) > 0 and False:
            index = cast(LMI, self._index)
            for i in range(len(deleted_ids)):
                id = deleted_ids[i]
                # Never added this id to hnsw, so we can safely ignore it for deletions
                if id not in self._id_to_label:
                    continue
                label = self._id_to_label[id]

                index.mark_deleted(label)
                del self._id_to_label[id]
                del self._label_to_id[label]
                del self._id_to_seq_id[id]

        # Writing in context of LMI means adding points to internal dataset
        # This dataset is held in memory, so it is not very efficient and for large dataset unusable
        # Will need to refactor it later
        if len(written_ids) > 0:
            self._ensure_index(batch.add_count, len(vectors_to_write[0]))

            next_label = self._total_elements_added + 1
            for i in range(len(written_ids)):
                if written_ids[i] not in self._id_to_label:
                    labels_to_write[i] = next_label
                    next_label += 1
                else:
                    labels_to_write[i] = self._id_to_label[written_ids[i]]

            index = cast(LMI, self._index)

            # First, update the index
            index.add_items(vectors_to_write, labels_to_write)

            # If that succeeds, update the mappings
            for i, id in enumerate(written_ids):
                self._id_to_seq_id[id] = batch.get_record(id)["seq_id"]
                self._seq_id_to_id = {value: key for key, value in self._id_to_seq_id.items()}
                self._id_to_label[id] = labels_to_write[i]
                self._label_to_id[labels_to_write[i]] = id

            # If that succeeds, update the total count
            self._total_elements_added += batch.add_count

            # If that succeeds, finally the seq ID
            self._max_seq_id = batch.max_seq_id


    def _write_records(self, records: Sequence[EmbeddingRecord]) -> None:
        """Add a batch of embeddings to the index"""
        if not self._running:
            raise RuntimeError("Cannot add embeddings to stopped component")

        # Avoid all sorts of potential problems by ensuring single-threaded access
        with WriteRWLock(self._lock):
            batch = Batch()

            for record in records:
                self._max_seq_id = max(self._max_seq_id, record["seq_id"])
                id = record["id"]
                op = record["operation"]
                label = self._id_to_label.get(id, None)

                if op == Operation.DELETE:
                    if label:
                        batch.apply(record)
                    else:
                        logger.warning(f"Delete of nonexisting embedding ID: {id}")

                elif op == Operation.UPDATE:
                    if record["embedding"] is not None:
                        if label is not None:
                            batch.apply(record)
                        else:
                            logger.warning(
                                f"Update of nonexisting embedding ID: {record['id']}"
                            )
                elif op == Operation.ADD:
                    if not label:
                        batch.apply(record, False)
                    else:
                        logger.warning(f"Add of existing embedding ID: {id}")
                elif op == Operation.UPSERT:
                    batch.apply(record, label is not None)

            self._apply_batch(batch)

    @override
    def delete(self) -> None:
        raise NotImplementedError()
