import multiprocessing
import re
from typing import Any, Callable, Dict, Union, List
import ast

from chromadb.types import Metadata
from chromadb.li_index.search.li.clustering import algorithms


Validator = Callable[[Union[str, int, float]], bool]

param_validators: Dict[str, Validator] = {
    "lmi:space": lambda p: bool(re.match(r"^(l2|cosine|ip)$", str(p))),
    "lmi:num_threads": lambda p: isinstance(p, int),
    "lmi:clustering_algorithms": lambda p: isinstance(p, str) and p != "",
    "lmi:epochs": lambda p: isinstance(p, str) and p != "",
    "lmi:model_types": lambda p: isinstance(p, str) and p != "",
    "lmi:lrs": lambda p: isinstance(p, str) and p != "",
    "lmi:n_categories": lambda p: isinstance(p, str) and p != "",
}

# Extra params used for persistent lmi
persistent_param_validators: Dict[str, Validator] = {
    "lmi:batch_size": lambda p: isinstance(p, int) and p > 2,
    "lmi:sync_threshold": lambda p: isinstance(p, int) and p > 2,
}


class Params:
    @staticmethod
    def _select(metadata: Metadata) -> Dict[str, Any]:
        segment_metadata = {}
        for param, value in metadata.items():
            if param.startswith("lmi:"):
                segment_metadata[param] = value
        return segment_metadata

    @staticmethod
    def _validate(metadata: Dict[str, Any], validators: Dict[str, Validator]) -> None:
        """Validates the metadata"""
        # Validate it
        for param, value in metadata.items():
            if param not in validators:
                raise ValueError(f"Unknown LMI parameter: {param}")
            if not validators[param](value):
                raise ValueError(f"Invalid value for LMI parameter: {param} = {value}")


class LMIParams(Params):
    space: str
    num_threads: int
    resize_factor: float

    def __init__(self, metadata: Metadata):
        metadata = metadata or {}
        self.space = metadata.get("lmi:space", "cosine")
        self.clustering_algorithms = metadata.get("lmi:clustering_algorithms", [algorithms['faiss_kmeans']])
        self.epochs = ast.literal_eval(metadata.get("lmi:epochs", "[200]"))
        self.model_types = ast.literal_eval(metadata.get("lmi:model_types", "['MLP']"))
        self.lrs = ast.literal_eval(metadata.get("lmi:lrs", "[0.01]"))
        self.n_categories = ast.literal_eval(metadata.get("lmi:n_categories", "[2, 2]"))
        self.num_threads = int(
            metadata.get("lmi:num_threads", multiprocessing.cpu_count())
        )

    @staticmethod
    def extract(metadata: Metadata) -> Metadata:
        """Validate and return only the relevant lmi params"""
        segment_metadata = LMIParams._select(metadata)
        LMIParams._validate(segment_metadata, param_validators)
        return segment_metadata


class PersistentHnswParams(LMIParams):
    batch_size: int
    sync_threshold: int

    def __init__(self, metadata: Metadata):
        super().__init__(metadata)
        self.batch_size = int(metadata.get("lmi:batch_size", 100))
        self.sync_threshold = int(metadata.get("lmi:sync_threshold", 1000))

    @staticmethod
    def extract(metadata: Metadata) -> Metadata:
        """Returns only the relevant lmi params"""
        all_validators = {**param_validators, **persistent_param_validators}
        segment_metadata = PersistentHnswParams._select(metadata)
        PersistentHnswParams._validate(segment_metadata, all_validators)
        return segment_metadata
