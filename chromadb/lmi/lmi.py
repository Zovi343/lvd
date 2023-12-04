from chromadb.lmi import ChromaIndex
from chromadb.lmi.index.LearnedIndex import LearnedIndex
from chromadb.lmi.index.BuildConfiguration import BuildConfiguration
from chromadb.lmi.index.clustering import ClusteringAlgorithm
from chromadb.lmi.index.clustering import algorithms
from typing import List
import numpy as np

class LMI(LearnedIndex, ChromaIndex):
    _build_config = None
    _dataset = None

    def __init__(self):
        super().__init__()
        _build_config = self._default_configuration()
        _dataset = np.array([])


    @staticmethod
    def _default_configuration():
        n_categories = [10, 10]

        return BuildConfiguration(
            [algorithms['faiss_kmeans']],
            [200],
            ['MLP'],
            [0.01],
            n_categories,
        )

    def add_items(self, data: List[List], ids=None, num_threads=-1, replace_deleted=False):
        data_converted = np.array(data)
        self._dataset = np.concatenate(self._dataset, data_converted)


    def init_index(self, max_elements, clustering_algorithms: List[ClusteringAlgorithm], epochs: [int], model: [str], learning_rate: [int], n_categories: [int],  is_persistent_index=False, persistence_location=None):
        if algorithms is not None and epochs is not None and model is not None and learning_rate is not None:
            self._build_config = BuildConfiguration(
                clustering_algorithms,
                epochs,
                model,
                learning_rate,
                n_categories,
            )

