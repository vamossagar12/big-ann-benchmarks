import os
import scann
import numpy as np
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS

class Scann(BaseANN):
    def __init__(self, metric, index_params):
        self.name = "Scann"
        if index_params.get("dist") is None:
            print("Error: missing parameter dist")
            return
        self.searcher_type = index_params.get("searcher_type")
        self.dist = index_params.get("dist")
        if self.dist == "dot_product":
            self.spherical = True
        else:
            self.spherical = False
        self._metric = metric
        self.dims_per_block = 2 # Recommended value for dimensions per block
        if index_params.get("n_leaves") is not None:
            self.n_leaves = index_params.get("n_leaves")
        if index_params.get("avq_threshold") is not None:
            self.avq_threshold = index_params.get("avq_threshold")

    def track(self):
        return "T1"

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        if self.spherical:
            ds[np.linalg.norm(dataset, axis=1) == 0] = 1.0 / np.sqrt(ds.shape[1])
            ds /= np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        else:
            self.spherical = False
        if len(ds) < 20000:
            self.searcher = scann.scann_ops_pybind.builder(ds, 10, self.dist)\
                .score_brute_force()
        elif len(ds) < 100000:
            self.searcher = scann.scann_ops_pybind.builder(ds, 10, self.dist)\
                .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)\
                .build()
        else:
            self.searcher = scann.scann_ops_pybind.builder(ds, 10, self.dist)\
                .tree(self.n_leaves, 1, training_sample_size=len(ds), spherical=self.spherical, quantize_centroids=True)\
                .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)\
                .build()

    def query(self, X, k):
        self.res = self.searcher.search_batched_parallel(X, k, self.reorder, self.leaves_to_search)

    def index_name(self):
        index_name = f"scann.{self.searcher_type}_spherical{self.spherical}"
        if self.n_leaves is not None:
            index_name += f"_n_leaves{self.n_leaves}"
        if self.avq_threshold is not None:
            index_name += f"_avq_threshold{self.avq_threshold}"
        return index_name

    def get_searcher_assets(self):
        searcher_assets = ["dataset.npy", "datapoint_to_token.npy", "hashed_dataset.npy", "int8_dataset.npy", "int8_multipliers.npy", "dp_norms.npy"]
        return searcher_assets

    def index_files_to_store(self, dataset):
        return [self.create_index_dir(DATASETS[dataset]()), self.index_name(), self.get_searcher_assets()]

    def create_index_dir(self, dataset):
        index_dir = os.path.join(os.getcwd(), "data", "indices")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, "T1")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.__str__())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, dataset.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.index_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return index_dir

    def load_index(self, dataset):
        index_dir = self.create_index_dir(DATASETS[dataset]())
        if not (os.path.exists(index_dir)):
            return False
        self.searcher = scann.scann_ops_pybind.load_searcher(index_dir)

    def range_query(self, X, radius):
        k = 100
        multiplier = 10
        result = list()
        search_further = list()
        n_res = 0
        n, d = X.shape
        # First run a batch search to find vectors which need to be searched further
        ids, dists = self.searcher.search_batched_parallel(X, k, self.reorder, self.leaves_to_search)
        for index in range(len(dists)):
            if dists[index][k - 1] < radius:
                search_further.append(index)
                result.append(())
            else:
                # This vector doesn't need further searching.
                # Store the indexes and distances in the results list as a tuple
                result.append((ids[index], dists[index]))
                n_res += len(ids[index])

        # Keep searching further unless there is not vector which has it's k-nn distance > radius
        while len(search_further) > 0:
            # Get a subset of vectors which need further searching.
            vectors_to_search_further = X[search_further]
            new_search_further = list()
            # move k further
            k *= multiplier
            for idx in search_further:
                ids, dists = self.searcher.search_batched_parallel(vectors_to_search_further, k, self.reorder, self.leaves_to_search)
                for i, dist in enumerate(dists):
                    if dist[k - 1] < radius:
                        # This vector needs further searching.
                        # Storing the idx for future queries.
                        new_search_further.append(idx)
                    else:
                        # This vector doesn't need further searching.
                        # Store the indexes and distances in the results list as a tuple
                        actual_index = search_further[i]
                        result[actual_index] = (ids[i], dist)
                        n_res += len(dist)
            search_further.clear()
            search_further.extend(new_search_further)

        lims = np.zeros(n + 1, int)
        I = np.empty(n_res, int)
        D = np.empty(n_res, 'float')

        pointer = 0
        for i in range(n):
            ids, dists = result[i]
            lims[i + 1] = lims[i] + len(ids)
            ind = range(pointer, pointer + len(ids))
            np.put(I, ind, ids)
            np.put(D, ind, dists)
            pointer += len(ids)
        self.res = lims, I, D

    def set_query_arguments(self, query_args):
        self._query_args = query_args
        if query_args.get("reorder") is not None:
            self.reorder = query_args.get("reorder")
        if query_args.get("leaves_to_search") is not None:
            self.leaves_to_search = query_args.get("leaves_to_search")

    def get_results(self):
        ids, dists = self.res
        return ids

    def get_range_results(self):
        return self.res
