"""
Code for finding K Nearest Neighbors in a dataset.

Please complete the function `calc_k_nearest_neighbors`

Examples
--------
See test_knn.py for example inputs and expected outputs.

To verify correctness of your implementation, you can execute:

$ python -m doctest test_knn.py
"""

import numpy as np


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    """Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    """

    # Unpack to get number of examples (N), features (F), and queries (Q)
    N, F = data_NF.shape
    Q, F2 = query_QF.shape
    assert F == F2

    K = int(K)
    if K < 1:
        raise ValueError("Invalid number of neighbors (K). Too small.")
    if K > N:
        raise ValueError("Invalid number of neighbors (K). Too large.")

    neighb_QKF = np.zeros((Q, K, F))  # placeholder to reserve the right shape

    for q in range(Q):
        query_F = query_QF[q]
        distances = np.zeros(N)
        for n, data_F in enumerate(data_NF):
            distance_F = query_F - data_F
            distance_F = distance_F * distance_F
            distance_1 = np.sqrt(np.sum(distance_F))
            distances[n] = distance_1

        # stable sort so ties result in selection of first appearing feature
        ids_NF = np.argsort(distances, kind="stable")
        neighb_QKF[q] = data_NF[ids_NF[:K]]

    return neighb_QKF
