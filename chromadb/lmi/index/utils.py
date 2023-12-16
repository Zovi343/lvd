import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def pairwise_cosine(x, y):
    # TODOL  find out what is category_L1 and category_L2
    y = y.drop(['category_L1'], axis=1) # , 'category_L2'
    return 1-cosine_similarity(x, y)


def pairwise_cosine_threshold(
    x: npt.NDArray[np.float32],
    y: pd.DataFrame,
    threshold: npt.NDArray[np.float64],
    cat_idxs: npt.NDArray[np.int32],
    k: int = 10,
):
    s = time.time()
    # TODO:  find out what is category_L1 and category_L2
    y = y.drop(['category_L1'], axis=1) # , 'category_L2'
    result = 1-cosine_similarity(x, y)
    t_pure_seq_search = time.time() - s
    # create an array of consisten shapes
    #print(result.shape)
    #print(threshold.shape, threshold[cat_idxs].shape)
    thresh_consistent = np.repeat(threshold[cat_idxs, np.newaxis], result.shape[1], 1)
    relevant_dists = np.where(result < thresh_consistent)
    # filter the relevant object ids
    try:
        relevant_object_ids = np.unique(relevant_dists[1])
        max_idx = relevant_object_ids.shape[0]
        if max_idx == 0:
            return None, t_pure_seq_search
    except ValueError:
        # There is no distance below the threshold, we can return
        return None, t_pure_seq_search
    max_idx = max_idx if max_idx > k else k
    # output array filled with some large value
    output_arr = np.full(shape=(result.shape[0], max_idx), fill_value=10_000, dtype=float)
    #index_df = pd.DataFrame(relevant_dists[0])
    #index_df = pd.DataFrame(relevant_dists[0], relevant_dists[1]).reset_index()
    # create indexes to store the relevant distances
    #index_df['mapping'] = index_df.groupby('index').ngroup()
    mapping = dict(zip(relevant_object_ids, np.arange(relevant_object_ids.shape[0])))
    # tried also with np.vectorize, wasn't faster
    output_arr_2nd_dim = np.array([mapping[x] for x in relevant_dists[1]])
    to_be_added = result[relevant_dists[0], relevant_dists[1]]
    # populate the output array
    output_arr[relevant_dists[0], output_arr_2nd_dim] = to_be_added
    return output_arr, relevant_object_ids, t_pure_seq_search

def save_as_pickle(filename: str, obj):
    """
    Saves an object as a pickle file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    obj: object
        The object to save.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
