"""
Test Cases Part 1
-----------------
# Focus: make_train_and_test_row_ids_for_n_fold_cv

# Let's try N=4 and K=3, where K doesn't divide evenly into N
>>> N = 4
>>> K = 3
>>> tr_ids_per_fold_K, te_ids_per_fold_K = (
...     make_train_and_test_row_ids_for_n_fold_cv(N, K))

# There should be K=3 entries in each returned list
>>> len(tr_ids_per_fold_K)
3

# Measure size of test set for each of the K folds
# K=3 doesn't evenly divide N=4, so we'll have two folds of 1 and one fold of 2
>>> np.sort([len(te) for te in te_ids_per_fold_K])
array([1, 1, 2])

# Measure size of train set for each of the K folds
# One pair of folds will have 1+1 = 2, two pairs will have 2+1 = 3
>>> np.sort([len(tr) for tr in tr_ids_per_fold_K])
array([2, 3, 3])

# Each row_id in {0, 1, ..., N-1} should appear in test exactly ONCE
>>> np.sort(np.hstack([te_ids_per_fold_K[kk] for kk in range(K)]))
array([0, 1, 2, 3])

# Each row_id in {0, 1, ..., N-1} should appear in train exactly K-1 times
>>> np.sort(np.hstack([tr_ids_per_fold_K[kk] for kk in range(K)]))
array([0, 0, 1, 1, 2, 2, 3, 3])


Test Cases Part 2
-----------------
# Focus: train_models_and_calc_scores_for_n_fold_cv

# Create simple dataset of N examples where y given x
# is perfectly explained by a linear regression model
>>> N = 24 # num examples
>>> K = 7  # num folds
>>> x_N3 = np.random.RandomState(0).rand(N, 3)
>>> y_N = np.matmul(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
>>> y_N.shape
(24,)

# Use linear regression (LR) implementation from sklearn
# as the base model that we fit via cross-validation.
>>> import sklearn.linear_model
>>> my_regr = sklearn.linear_model.LinearRegression()

# Fit the LR model on each of the K train splits
# and also measure the train and test error of each split
>>> err_tr_K, err_te_K = train_models_and_calc_scores_for_n_fold_cv(
...                 my_regr, x_N3, y_N, n_folds=K, random_state=0)

# Training error should be indistiguishable from zero
# because we have enough data (N > F+1) and we know true model is linear
>>> np.array2string(err_tr_K, precision=3, suppress_small=True)
'[0. 0. 0. 0. 0. 0. 0.]'

# Testing error should be indistinguishable from zero
# because we've fit the true model well
>>> np.array2string(err_te_K, precision=3, suppress_small=True)
'[0. 0. 0. 0. 0. 0. 0.]'
"""

import numpy as np
from performance_metrics import calc_root_mean_squared_error


def train_models_and_calc_scores_for_n_fold_cv(
    estimator, x_NF, y_N, n_folds=3, random_state=0
):
    """Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    """
    N, F = x_NF.shape
    assert (N,) == y_N.shape
    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float32)

    train_ids_per_fold, test_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(
        N, n_folds, random_state
    )

    for f in range(n_folds):
        # Gather training examples and response
        # Let S be the number of training examples
        x_SF = x_NF[train_ids_per_fold[f]]
        y_S = y_N[train_ids_per_fold[f]]
        (S1, F1) = x_SF.shape
        (S2,) = y_S.shape
        assert F1 == F
        assert S1 == S2  # as many examples as outputs

        estimator.fit(x_SF, y_S)

        # Training fit
        yhat_S = estimator.predict(x_SF)
        (S3,) = yhat_S.shape
        assert S3 == S2
        # Gather test examples and response
        # Let T be the number of test examples

        x_TF = x_NF[test_ids_per_fold[f]]
        y_T = y_N[test_ids_per_fold[f]]

        (T1, F2) = x_TF.shape
        (T2,) = y_T.shape
        assert T1 == T2
        assert F2 == F

        # Test fit
        yhat_T = estimator.predict(x_TF)
        (T3,) = yhat_T.shape
        assert T3 == T2

        # Calculate errors
        train_error = calc_root_mean_squared_error(y_S, yhat_S)
        test_error = calc_root_mean_squared_error(y_T, yhat_T)

        train_error_per_fold[f] = train_error
        test_error_per_fold[f] = test_error

    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(n_examples=0, n_folds=3, random_state=0):
    """Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N
    """
    if hasattr(random_state, "rand"):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state  # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    train_ids_per_fold = list()
    test_ids_per_fold = list()

    N = n_examples
    random_order = random_state.permutation(N)

    test_start = 0
    subset_size = n_examples // n_folds
    for k in range(1, n_folds + 1):
        if k == n_folds:
            test_set = random_order[test_start:]
            train_set = random_order[0:test_start]

            assert len(test_set) + len(train_set) == N
            assert len(np.intersect1d(test_set, train_set)) == 0
            assert np.array_equal(
                np.union1d(test_set, train_set), [n for n in range(N)]
            )

            train_ids_per_fold.append(train_set)
            test_ids_per_fold.append(test_set)
        else:
            test_set = random_order[test_start : test_start + subset_size]

            train_set = np.concatenate(
                (random_order[0:test_start], random_order[test_start + subset_size :])
            )
            assert len(test_set) + len(train_set) == N
            assert len(np.intersect1d(test_set, train_set)) == 0
            assert np.array_equal(
                np.union1d(test_set, train_set), [n for n in range(N)]
            )

            train_ids_per_fold.append(train_set)
            test_ids_per_fold.append(test_set)

        test_start += subset_size

    return train_ids_per_fold, test_ids_per_fold
