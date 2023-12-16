import time
from collections import defaultdict
from itertools import product, takewhile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.utils.data
from chromadb.lmi.index.BuildConfiguration import BuildConfiguration
from chromadb.lmi.index.clustering import ClusteringAlgorithm
from chromadb.lmi.index.Logger import Logger
from chromadb.lmi.index.model import LIDataset, NeuralNetwork, data_X_to_torch
from chromadb.lmi.index.utils import pairwise_cosine, pairwise_cosine_threshold
from chromadb.lmi.attribtue_filtering.default_filtering import attribute_filtering
from tqdm import tqdm

torch.manual_seed(2023)
np.random.seed(2023)

EMPTY_VALUE = -1


class LearnedIndex(Logger):
    def __init__(self):
        self.root_model: Optional[NeuralNetwork] = None
        """Root model can be accessed after calling `build`."""

        self.internal_models: Dict[Tuple, NeuralNetwork] = {}
        """
        Dictionary mapping the path to the internal model.
        A path is padded with `EMPTY_VALUE` to the right to match the length of the longest path.
        """

        self.bucket_paths: List[Tuple] = []
        """List of paths to the buckets."""

    def search(
        self,
        data_navigation: pd.DataFrame,
        queries_navigation: npt.NDArray[np.float32],
        data_search: pd.DataFrame,
        queries_search: npt.NDArray[np.float32],
        data_prediction: npt.NDArray[np.int64],
        n_categories: List[int],
        n_buckets: int = 1,
        k: int = 10,
        use_threshold: bool = True,
        attribute_filter: Optional[npt.NDArray[np.uint32]] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], npt.NDArray, Dict[str, float]]:
        """Searches for `k` nearest neighbors for each query in `queries`.

        Implementation details:
        - The search is done in two steps:
            1. The order in which the queries visit the buckets is precomputed.
            2. The queries are then searched in the `n_buckets` most similar buckets.

        Parameters
        ----------
        data_navigation : pd.DataFrame
            Data used for navigation.
        queries_navigation : npt.NDArray[np.float32]
            Queries used for navigation.
        data_search : pd.DataFrame
            Data used for the sequential search.
        queries_search : npt.NDArray[np.float32]
            Queries used for the sequential search.
        data_prediction : npt.NDArray[np.int64]
            Predicted paths for each data point.
        n_categories : List[int]
            Number of categories for each level of the index.
        n_buckets : int, optional
            Number of most similar buckets to search in, by default 1
        k : int, optional
            Number of nearest neighbors to search for, by default 10
        use_threshold : bool, optional
            Whether to use the threshold distance to filter the objects, by default True

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray[np.uint32], Dict[str, float]]
            Array of shape (queries_search.shape[0], k) with distances to nearest neighbors for each query,
            array of shape (queries_search.shape[0], k) with nearest neighbors for each query,
            dictionary with measured times.
        """
        measured_time = defaultdict(float)

        s = time.time()

        anns_final = None
        dists_final = None

        self.logger.debug(f"Precomputing bucket order")
        s_ = time.time()
        bucket_order, measured_time["inference"] = self._precompute_bucket_order(
            queries_navigation=queries_navigation,
            n_buckets=n_buckets,
            n_categories=n_categories,
        )
        self.logger.info(f"Precompute bucket order time: {time.time() - s_}")

        # Search in the `n_buckets` most similar buckets
        for bucket_order_idx in range(n_buckets):
            if bucket_order_idx != 0 and use_threshold:
                assert dists_final is not None
                threshold_dist = dists_final.max(axis=1)
            else:
                threshold_dist = None

            self.logger.debug(
                f"Searching in bucket {bucket_order_idx + 1} out of {n_buckets}"
            )
            t = time.time()
            (
                dists,
                anns,
                t_all,
                t_seq_search,
                t_sort,
            ) = self._search_single_bucket(
                data_navigation=data_navigation,
                data_search=data_search,
                queries_search=queries_search,
                data_prediction=data_prediction,
                bucket_path=bucket_order[:, bucket_order_idx, :],
                threshold_dist=threshold_dist,
                n_levels=len(n_categories),
                attribute_filter=attribute_filter
            )
            self.logger.debug(f"Searched the bucket in: {time.time() - t}")

            measured_time["search_within_buckets"] += t_all
            measured_time["seq_search"] += t_seq_search
            measured_time["sort"] += t_sort

            self.logger.debug(f"Sorting the results")
            t = time.time()
            if anns_final is None:
                anns_final = anns
                dists_final = dists
            else:
                # stacks the results from the previous sorted anns and dists
                # *_final arrays now have shape (queries.shape[0], k*2)
                anns_final = np.hstack((anns_final, anns))
                dists_final = np.hstack((dists_final, dists))
                # gets the sorted indices of the stacked dists
                idx_sorted = dists_final.argsort(kind="stable", axis=1)[:, :k]
                # indexes the final arrays with the sorted indices
                # *_final arrays now have shape (queries.shape[0], k)
                idx = np.ogrid[tuple(map(slice, dists_final.shape))]
                idx[1] = idx_sorted
                dists_final = dists_final[tuple(idx)]
                anns_final = anns_final[tuple(idx)]

                assert (
                    anns_final.shape
                    == dists_final.shape
                    == (queries_search.shape[0], k)
                )
            self.logger.debug(f"Sorted the results in: {time.time() - t}")

        assert dists_final is not None
        assert anns_final is not None

        measured_time["search"] = time.time() - s

        return dists_final, anns_final, bucket_order, measured_time

    def _precompute_bucket_order(
        self,
        queries_navigation: npt.NDArray[np.float32],
        n_buckets: int,
        n_categories: List[int],
    ) -> Tuple[npt.NDArray[np.int32], float]:
        """
        Precomputes the order in which the queries visit the buckets.

        Implementation details:
        - When visiting an internal node, the paths to the nodes/buckets under that node
        are added to the priority queue.
        - When visiting a bucket, the path to the bucket is stored in `bucket_order`.
        - The priority queue is then sorted by the probability of the next node/bucket to visit.
        - The computation is done until `n_buckets` buckets are visited for each query.

        Parameters
        ----------
        queries_navigation : np.ndarray
            Queries used for navigation.
        n_buckets : int
            Number of most similar buckets to precompute the order for.
        n_categories : List[int]
            Number of categories for each level of the index.

        Returns
        -------
        Tuple[npt.NDArray[np.int32], float]
            Array of shape (queries_navigation.shape[0], n_buckets, len(n_categories))
            with the order in which the queries visit the buckets,
            total inference time.
        """

        n_queries = queries_navigation.shape[0]
        n_levels = len(n_categories)

        class PriorityQueue:
            """
            A priority queue storing probabilities and paths for the next nodes to visit for each query.

            The priority queue is realized by three numpy arrays:
            - `probability`: stores the probability of the next node/bucket to visit
            - `path`: stores the path to the next node/bucket
            - `length`: stores the current length of the queue for each query
            - `should_sort`: stores whether the queue associated with this query should be sorted

            Notes:
            - `total_n_buckets` is an upper bound for the queue length
            (equal to the total number of buckets in the index)
            """

            def __init__(self):
                total_n_buckets = np.prod(n_categories)
                self.probability: npt.NDArray[np.float32] = np.full(
                    (n_queries, total_n_buckets),
                    fill_value=EMPTY_VALUE,
                    dtype=np.float32,
                )
                self.path: npt.NDArray[np.int32] = np.full(
                    (n_queries, total_n_buckets, n_levels),
                    fill_value=EMPTY_VALUE,
                    dtype=np.int32,
                )
                self.length: npt.NDArray[np.int32] = np.zeros(n_queries, dtype=np.int32)

                self.should_sort: npt.NDArray[np.bool_] = np.full(
                    n_queries, fill_value=np.False_, dtype=np.bool_
                )

            def add(
                self,
                indices: npt.NDArray[np.int32],
                path: npt.NDArray[np.int32],
                probabilities: npt.NDArray[np.float32],
            ) -> None:
                """
                Adds a new node/bucket path to visit to the priority queue
                but only for queries specified by `indices`.
                """
                self.probability[indices, self.length[indices]] = probabilities
                self.path[indices, self.length[indices], :] = path
                self.should_sort[indices] = np.True_

                self.length[indices] += 1

            def pop(self, indices: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
                """Pops the next node/bucket path to visit for each query."""
                self.length[indices] -= 1

                return self.path[indices, self.length[indices], :]

            def sort(self) -> None:
                """
                Sorts the queues by the probability.
                A particular queue is sorted only if `should_sort` is `True`.

                Implementation details:
                Firstly, we obtain the indexes of the sorted probabilities.
                Then, we use these indexes to sort the probabilities and paths.
                Sorting of paths is done for each level separately.
                The whole process is repeated for each queue length separately.
                """
                for queue_length in np.unique(self.length):
                    if queue_length in {0, 1}:
                        continue

                    idxs_to_sort = np.where(
                        np.logical_and(
                            self.length == queue_length,
                            self.should_sort == np.True_,
                        )
                    )[0]

                    sorted_idxs = self.probability[
                        idxs_to_sort, :queue_length
                    ].argsort()

                    self.probability[idxs_to_sort, :queue_length] = np.take_along_axis(
                        self.probability[idxs_to_sort, :queue_length],
                        sorted_idxs,
                        axis=1,
                    )
                    for level_idx in range(n_levels):
                        self.path[
                            idxs_to_sort, :queue_length, level_idx
                        ] = np.take_along_axis(
                            self.path[idxs_to_sort, :queue_length, level_idx],
                            sorted_idxs,
                            axis=1,
                        )

                    self.should_sort[idxs_to_sort] = np.False_

        def visit_internal_nodes(
            all_query_idxs: npt.NDArray[np.int32],
            pq: PriorityQueue,
            path_to_visit: npt.NDArray[np.int32],
        ) -> float:
            """
            Visits the internal nodes specified by `paths`.
            Paths to the buckets under a specific internal node are then added into `pq`.
            Done for each possible path to an internal node separately.
            """
            total_inference_t = 0.0

            for path in self.internal_models.keys():
                path_idxs = LearnedIndex._filter_idxs(path_to_visit, path)
                query_idxs = all_query_idxs[path_idxs]
                if query_idxs.shape[0] == 0:
                    continue

                model = self.internal_models[path]

                s = time.time()
                probabilities, categories = model.predict_proba(
                    data_X_to_torch(queries_navigation[query_idxs])
                )
                total_inference_t += time.time() - s

                level = len(path) - path.count(EMPTY_VALUE)
                n_model_categories = categories.shape[1]

                for child_idx in range(n_model_categories):
                    child_paths = np.full(
                        (query_idxs.shape[0], n_levels),
                        fill_value=EMPTY_VALUE,
                        dtype=np.int32,
                    )
                    child_paths[:] = np.array(path)
                    child_paths[:, level] = categories[:, child_idx]

                    pq.add(
                        query_idxs,
                        child_paths,
                        probabilities[:, child_idx],
                    )

            return total_inference_t

        def visit_buckets(
            all_query_idxs: npt.NDArray[np.int32],
            path_to_visit: npt.NDArray[np.int32],
            bucket_order: npt.NDArray[np.int32],
            bucket_order_length: npt.NDArray[np.int32],
        ) -> None:
            """
            Visits the buckets specified by `paths`.
            The path to the bucket relevant to each query is then stored in `bucket_order`.
            Done for each possible bucket separately.
            """
            for path in self.bucket_paths:
                path_idxs = LearnedIndex._filter_idxs(path_to_visit, path)
                query_idxs = all_query_idxs[path_idxs]

                if query_idxs.shape[0] == 0:
                    continue

                bucket_order[query_idxs, bucket_order_length[query_idxs], :] = np.array(
                    path
                )
                bucket_order_length[query_idxs] += 1

        assert self.root_model is not None, "Model is not trained, call `build` first."

        total_inference_t = 0.0

        s = time.time()
        pred_l1_prob, pred_l1_paths = self.root_model.predict_proba(
            data_X_to_torch(queries_navigation)
        )
        total_inference_t += time.time() - s

        if n_levels == 1:
            bucket_order = np.full(
                (n_queries, n_buckets, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
            )
            bucket_order[:, :n_buckets, 0] = pred_l1_paths[:, :n_buckets]
            return bucket_order, total_inference_t

        pq = PriorityQueue()

        # Populates the priority queue with the first level of the index
        # * Relies on the fact that the pred_l1_categories and pred_l1_probs are sorted,
        # * therefore the priority queue does not need to be sorted after this for loop
        for l1_idx in reversed(range(n_categories[0])):
            l1_paths = np.full(
                (n_queries, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
            )
            l1_paths[:, 0] = pred_l1_paths[:, l1_idx]
            pq.add(np.arange(n_queries), l1_paths, pred_l1_prob[:, l1_idx])

        bucket_order = np.full(
            (n_queries, n_buckets, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
        )
        bucket_order_length = np.zeros(n_queries, dtype=np.int32)

        while not np.all(bucket_order_length == n_buckets):
            query_idxs = np.where(bucket_order_length < n_buckets)[0]
            path_to_visit = pq.pop(query_idxs)

            inference_t = visit_internal_nodes(query_idxs, pq, path_to_visit)
            visit_buckets(query_idxs, path_to_visit, bucket_order, bucket_order_length)

            total_inference_t += inference_t

            pq.sort()

        return bucket_order, total_inference_t

    def _search_single_bucket(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        queries_search: npt.NDArray[np.float32],
        data_prediction: npt.NDArray[np.int64],
        bucket_path: npt.NDArray[np.int32],
        k: int = 10,
        threshold_dist: Optional[npt.NDArray[np.float64]] = None,
        n_levels: int = 1,
        attribute_filter: Optional[npt.NDArray[np.uint32]] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], float, float, float]:
        s_all = time.time()

        n_queries = queries_search.shape[0]
        nns = np.zeros((n_queries, k), dtype=np.uint32)
        dists = np.full((n_queries, k), fill_value=float("inf"), dtype=float)

        possible_bucket_paths = []
        for level_idx_to_search in range(n_levels):
            data_navigation[f"category_L{level_idx_to_search + 1}"] = data_prediction[
                :, level_idx_to_search
            ]
            possible_bucket_paths.append(f"category_L{level_idx_to_search + 1}")

        t_seq_search = 0.0
        t_sort = 0.0

        for path, g in tqdm(data_navigation.groupby(possible_bucket_paths)):
            bucket_obj_indexes = g.index

            relevant_query_idxs = LearnedIndex._filter_idxs(bucket_path, path)

            if bucket_obj_indexes.shape[0] != 0 and relevant_query_idxs.shape[0] != 0:
                s = time.time()

                if threshold_dist is not None:
                    seq_search_dists = pairwise_cosine_threshold(
                        queries_search[relevant_query_idxs],
                        data_search.loc[bucket_obj_indexes],
                        threshold_dist,
                        relevant_query_idxs,
                        k,
                    )
                    if seq_search_dists[0] is None:
                        # There is no distance below the threshold, we can continue
                        continue
                    else:
                        # seq_search_dists[1] contains the indexes of the relevant objects
                        bucket_obj_indexes = bucket_obj_indexes[seq_search_dists[1]]
                        seq_search_dists = seq_search_dists[0]
                else:
                    seq_search_dists = pairwise_cosine(
                        queries_search[relevant_query_idxs],
                        data_search.loc[bucket_obj_indexes],
                    )
                t_seq_search += time.time() - s
                s = time.time()
                ann_relative = seq_search_dists.argsort(kind="quicksort")

                # Perform bucket level attribute filtering
                if attribute_filter is not None:
                    ann_relative = attribute_filtering(ann_relative, attribute_filter, bucket_obj_indexes)

                ann_relative = ann_relative[
                    :,
                    : k if k < seq_search_dists.shape[1] else seq_search_dists.shape[1],
                ]
                t_sort += time.time() - s
                if bucket_obj_indexes.shape[0] < k:
                    # pad to `k` if needed
                    pad_needed = (k - bucket_obj_indexes.shape[0]) // 2 + 1
                    bucket_obj_indexes = np.pad(
                        np.array(bucket_obj_indexes), pad_needed, "edge"
                    )[:k]
                    ann_relative = np.pad(ann_relative[0], pad_needed, "edge")[
                        :k
                    ].reshape(1, -1)
                    seq_search_dists = np.pad(seq_search_dists[0], pad_needed, "edge")[
                        :k
                    ].reshape(1, -1)
                    _, i = np.unique(seq_search_dists, return_index=True)
                    duplicates_i = np.setdiff1d(np.arange(k), i)
                    # assign a large number such that the duplicated value gets replaced
                    seq_search_dists[0][duplicates_i] = 10_000

                nns[relevant_query_idxs] = np.array(bucket_obj_indexes)[ann_relative]
                dists[relevant_query_idxs] = np.take_along_axis(
                    seq_search_dists, ann_relative, axis=1
                )

        return dists, nns, time.time() - s_all, t_seq_search, t_sort

    def build(
        self,
        data: pd.DataFrame,
        config: BuildConfiguration,
    ) -> Tuple[npt.NDArray[np.int64], int, float, float]:
        """
        Builds the index.

        Parameters
        ----------
        data : pd.DataFrame
            Data to build the index on.
        config : BuildConfiguration
            Configuration for the training.

        Returns
        -------
        Tuple[npt.NDArray[np.int64], int, float, float]
            An array of shape (data.shape[0], len(config.n_levels)) with predicted paths for each data point,
            number of buckets, time it took to build the index, time it took to cluster the data.
        """
        s = time.time()

        n_levels = config.n_levels

        # Where should the training data be placed with respect to each level
        data_prediction: npt.NDArray[np.int64] = np.full(
            (data.shape[0], n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
        )

        self.logger.debug("Training the root model.")
        self.root_model, root_cluster_t = self._train_model(
            data,
            **config.level_configurations[0],
        )
        data_prediction[:, 0] = self.root_model.predict(data_X_to_torch(data))
        self.logger.debug(f"Trained the root model in {time.time()-s:.2f}s.")

        if n_levels == 1:
            for bucket_index in range(len(np.unique(data_prediction[:, 0]))):
                self.bucket_paths.append((bucket_index,))

            return (
                data_prediction,
                len(self.bucket_paths),
                time.time() - s,
                root_cluster_t,
            )

        self.logger.debug(f"Training {config.n_categories[:-1]} internal models.")
        s_internal = time.time()
        internal_cluster_t = self._train_internal_models(
            data,
            data_prediction,
            config,
        )
        self.logger.debug(
            f"Trained {config.n_categories[:-1]} internal models in {time.time()-s_internal:.2f}s."
        )

        return (
            data_prediction,
            len(self.bucket_paths),
            time.time() - s,
            root_cluster_t + internal_cluster_t,
        )

    def _train_model(
        self,
        data: pd.DataFrame,
        # ModelParameters
        clustering_algorithm: ClusteringAlgorithm,
        model_type: str,
        epochs: int,
        lr: float,
        n_categories: int,
    ) -> Tuple[NeuralNetwork, float]:
        """
        Trains a single model.
        The model is trained until it predicts the correct number of categories.
        The same number of epochs is used for each training iteration.

        Parameters
        ----------
        data : pd.DataFrame
            Data to train the model on.
        clustering_algorithm : ClusteringAlgorithm
            Clustering algorithm to use.
        model_type : str
            Type of the model.
        epochs : int
            The minimal number of epochs to train the model for.
        lr : float
            Learning rate for the model.
        n_categories : int
            Number of categories to predict.

        Returns
        -------
        Tuple[NeuralNetwork, float]
            Trained model, time it took to cluster the data.

        Raises
        ------
        RuntimeError
            If the model does not converge after 1000 iterations
            (after training `epochs` epochs 1000 times).
        """
        _, labels, cluster_t = self._cluster(data, clustering_algorithm, n_categories)
        n_clusters = len(np.unique(labels))

        if n_clusters != n_categories:
            self.logger.debug(
                "Clustering algorithm did not return %d clusters, got %d.",
                n_categories,
                n_clusters,
            )
            self.logger.debug("Setting n_categories to %d.", n_clusters)
            n_categories = n_clusters

        train_loader = torch.utils.data.DataLoader(
            dataset=LIDataset(data, labels),
            batch_size=256,
            sampler=torch.utils.data.SubsetRandomSampler(data.index.values.tolist()),
        )
        torch_data = data_X_to_torch(data)

        model = NeuralNetwork(
            input_dim=data.shape[1],
            output_dim=n_categories,
            lr=lr,
            model_type=model_type,
        )
        is_trained = False

        iters = 0
        while not is_trained:
            model.train_batch(train_loader, epochs=epochs, logger=self.logger)
            predictions = model.predict(torch_data)
            iters += 1

            if iters > 1_000:
                raise RuntimeError("The model did not converge after 1000 iterations.")

            is_trained = len(np.unique(predictions)) == n_categories

        if iters > 1:
            self.logger.debug(
                f"Trained for {iters * epochs} epochs instead of {epochs}."
            )

        return model, cluster_t

    def _train_internal_models(
        self,
        data: pd.DataFrame,
        data_prediction: npt.NDArray[np.int64],
        config: BuildConfiguration,
    ) -> float:
        """
        Trains the internal models.

        ! The `data_prediction` array is modified in-place.

        Parameters
        ----------
        data : pd.DataFrame
            Data to train the models on.
        data_prediction : npt.NDArray[np.int64]
            Predicted paths for each data point.
        config : BuildConfiguration
            Configuration for the training.

        Returns
        -------
        float
            Time it took to cluster the data.
        """
        assert (
            self.root_model is not None
        ), "The root model is not trained, call `_train_root_model` first."

        overall_cluster_t = 0.0

        for level in range(1, config.n_levels):
            internal_node_paths = self._generate_internal_node_paths(
                level, config.n_levels, config.n_categories
            )

            for path in internal_node_paths:
                data_idxs = LearnedIndex._filter_idxs(data_prediction, path)
                assert (
                    data_idxs.shape[0] != 0
                ), "There are no data points associated with the given path."

                # +1 as the data is indexed from 1
                training_data = data.loc[data_idxs + 1]

                # The subset needs to be reindexed; otherwise, the object accesses are invalid.
                original_pd_indices = training_data.index.values
                training_data = training_data.set_index(
                    pd.Index(range(1, training_data.shape[0] + 1))
                )

                model, cluster_t = self._train_model(
                    training_data,
                    **config.level_configurations[level],
                )
                self.internal_models[path] = model

                overall_cluster_t += cluster_t

                # Restore back to the original indices
                training_data = training_data.set_index(
                    pd.Index(original_pd_indices.tolist())
                )

                predictions = model.predict(data_X_to_torch(training_data))

                # original_pd_indices-1 as data is indexed from 1
                # level as we are predicting the next level but the indexing is 0-based
                data_prediction[original_pd_indices - 1, level] = predictions

                if level == config.n_levels - 1:
                    for bucket_index in range(len(np.unique(predictions))):
                        self.bucket_paths.append(path[:-1] + (bucket_index,))

        return overall_cluster_t

    def _cluster(
        self,
        data: pd.DataFrame,
        clustering_algorithm: ClusteringAlgorithm,
        n_clusters: int,
    ) -> Tuple[Optional[Any], npt.NDArray[np.int32], float]:
        s = time.time()

        if data.shape[0] < 2:
            return None, np.array([0] * data.shape[0]), time.time() - s

        if data.shape[0] < n_clusters:
            n_clusters = data.shape[0] // 5
            if n_clusters < 2:
                n_clusters = 2

        clustering_object, labels = clustering_algorithm(
            np.array(data),
            n_clusters,
            None,
        )

        return clustering_object, labels, time.time() - s

    def _serialize_path(self, path: Tuple) -> str:
        """
        Serializes the path to a string.

        Example:
        >>> self._serialize_path((1, 2, -1, -1))
        "1.2"
        """
        valid_path = takewhile(lambda x: x != EMPTY_VALUE, path)

        return ".".join(map(str, valid_path))

    def _deserialize_path(self, path: str, n_levels: int) -> Tuple:
        """
        Deserializes the path from a string.

        Example:
        >>> self._deserialize_path("1.2", 4)
        (1, 2, -1, -1)
        """
        levels = path.split(".")

        return tuple(list(map(int, levels)) + [EMPTY_VALUE] * (n_levels - len(levels)))

    def _generate_internal_node_paths(
        self, level: int, n_levels: int, n_categories: List[int]
    ) -> List[Tuple]:
        """Generates all possible paths to internal nodes at the given `level`.

        Parameters
        ----------
        level : int
            Desired level of the internal nodes.
        n_levels : int
            Total number of levels in the index.
        n_categories : List[int]
            Number of categories for each level of the index.

        Returns
        -------
        List[Tuple]
            List of all possible paths to internal nodes at the given `level`.
        """
        path_combinations = [range(n_categories[l]) for l in range(level)]
        padding = [[EMPTY_VALUE]] * (n_levels - level)

        return list(product(*path_combinations, *padding))

    @staticmethod
    def _filter_idxs(
        paths: npt.NDArray[Union[np.int32, np.int64]], path: Tuple
    ) -> npt.NDArray[np.int32]:
        """Returns the indexes of `paths` that match the given `path`."""
        return np.where(np.all(paths == np.array(path), axis=1))[0]
