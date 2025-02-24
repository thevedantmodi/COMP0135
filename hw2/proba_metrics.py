"""
proba_metrics.py

Provides implementation of common metrics for assessing a binary classifier's
*probabilistic* predictions against true binary labels, including:

* `calc_mean_BCE_from_probas`
Compute binary cross entropy (BCE) from probability predictions.

* `calc_mean_BCE_from_scores`
Compute binary cross entropy (BCE) from scores.
Scores here are the real-value inputs to sigmoid function.
Computing diretly from scores can be more numerically stable.


Test Cases: calc_mean_BCE_from_probas
-------------------------------------
>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])

# Try perfect predictions
>>> perfect_proba1_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
>>> perfect_bce = calc_mean_BCE_from_probas(
...     ytrue_N, perfect_proba1_N)
>>> print("%.4f" % perfect_bce)
0.0000

# Try some confident scores
>>> good_proba1_N = np.asarray([0.01, 0.05, 0.1, 0.1, 0.9, 0.9, 0.95, 0.97])
>>> good_bce = calc_mean_BCE_from_probas(
...     ytrue_N, good_proba1_N)
>>> print("%.4f" % good_bce)
0.1018

# Try some decent but underconfident scores
>>> ok_proba1_N = np.asarray([0.3, 0.4, 0.46, 0.47, 0.5, 0.6, 0.7, 0.71])
>>> ok_bce = calc_mean_BCE_from_probas(
...     ytrue_N, ok_proba1_N)
>>> print("%.4f" % ok_bce)
0.7253

# Try some mistakes that are way over confident
>>> bad_pr1_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
>>> bad = calc_mean_BCE_from_probas(ytrue_N, bad_pr1_N)
>>> print("%.4f" % bad)
17.4403

# Try empty
>>> empty_bce = calc_mean_BCE_from_probas([], [])
>>> np.allclose(0.0, empty_bce)
True


Test Cases for calc_mean_BCE_from_scores
----------------------------------------
>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])

# Try some good scores for these 8 points.
>>> good_scores_N = np.asarray([-4., -3., -2., -1., 1, 2., 3., 4.])
>>> good_bce = calc_mean_BCE_from_scores(ytrue_N, good_scores_N)
>>> print("%.6f" % (good_bce))
0.182835

# Try some near-perfect scores for these 8 points.
>>> perfect_scores_N = np.asarray([-9., -9., -9,  -9., 9., 9., 9., 9.])
>>> perfect_bce = calc_mean_BCE_from_scores(ytrue_N, perfect_scores_N)
>>> print("%.6f" % (perfect_bce))
0.000178

# Check same computation with "probas". Should match exactly.
>>> perfect_bce2 = calc_mean_BCE_from_probas(
...     ytrue_N, sigmoid(perfect_scores_N))
>>> print("%.6f" % (perfect_bce2))
0.000178

# Try some extreme scores (should get worse BCE with more extreme scores)
>>> ans = calc_mean_BCE_from_scores([0], [+99.])
>>> print("%.6f" % (ans))
142.826809
>>> ans = calc_mean_BCE_from_scores([0], [+999.])
>>> print("%.6f" % (ans))
1441.252346
>>> ans = calc_mean_BCE_from_scores([0], [+9999.])
>>> print("%.6f" % (ans))
14425.507714

# Try with "probas": sigmoid saturates! using scores is *better*
>>> a = calc_mean_BCE_from_probas([0], sigmoid([+999.]))
>>> print("%.6f" % (a))
46.508147
>>> a = calc_mean_BCE_from_probas([0], sigmoid([+9999.]))
>>> print("%.6f" % (a))
46.508147

# Try empty
>>> empty_bce = calc_mean_BCE_from_scores([], [])
>>> np.allclose(0.0, empty_bce)
True
"""

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logsumexp as scipy_logsumexp


def calc_mean_BCE_from_probas(ytrue_N, yproba1_N):
    """Compute average cross entropy for given binary classifier's predictions

    Consumes probabilities ("probas"), values between 0.0 and 1.0.

    Computing BCE uses *base-2* logarithms, so the resulting number is a valid
    upper bound of the zero-one loss (aka error rate) when we threshold at 0.5.

    Notes
    -----
    Given a binary label $y_n \in \{0, 1}$ and a probability $p_n \in (0,1)$,
    we define binary cross entropy as:
    $$
        BCE(y_n, p_n) = - y_n \log_2 p_n - (1-y_n) \log_2 (1-p_n)
    $$
    Given $N$ labels and their predicted probas, we define the mean BCE as:
    $$
        mean_BCE(y, p) = \frac{1}{N} \sum_{n=1}^N BCE(y_n, p_n)
    $$

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
        All values must be either 0 or 1. Will be cast to int dtype.
    yproba1_N : 1D array, shape (n_examples,) = (N,)
        Each entry is probability the corresponding example has positive label.
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.
        All values must be within the interval 0.0 to 1.0, inclusive.
        Will be truncated to (eps, 1 - eps) to keep log values from extremes,
        with small value eps equal to 10^{-14}.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples
    """
    # Cast labels to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = int(ytrue_N.size)
    if N == 0:
        return 0.0
    assert np.min(ytrue_N) >= 0
    assert np.max(ytrue_N) <= 1
    # Cast probas to float and be sure we're between zero and one
    yproba1_N = np.asarray(yproba1_N, dtype=np.float64)  # dont touch
    yproba1_N = np.maximum(1e-14, np.minimum(1 - 1e-14, yproba1_N))  # dont touch

    # compute BCE. Be sure to handle empty input lists properly
    bce = np.mean(
        -ytrue_N * np.log2(yproba1_N) - (1 - ytrue_N) * np.log2(1 - yproba1_N)
    )

    return bce


def calc_mean_BCE_from_scores(ytrue_N, scores_N):
    """Compute average cross entropy given binary classifier's scores.

    Consumes "scores", real values between (-np.inf, np.inf)
    This code computes the BCE directly from these scores,
    conceptually doing the two steps of
    (1) turning scores into probabilities via the logistic sigmoid
    (2) computing BCE from the probabilities in step 1.

    In practice, we compute BCE directly from scores using an implementation
    that avoids the numerical issues of the sigmoid (saturation, if underflows
    to zero would become negative infinity). This clever implementation uses
    the "logsumexp" trick.

    Computing BCE uses *base-2* logarithms, so the resulting number is a validc

    upper bound of mean zero-one loss (aka error rate) when threshold = 0.5.

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    scores_N : 1D array, shape (n_examples,) = (N,)
        One entry per example in current dataset.
        Each entry is a real value, could be between -infinity and +infinity.
        Large negative values indicate strong probability of label y=0.
        Zero values indicate probability of 0.5.
        Large positive values indicate strong probability of label y=1.
        Needs to be same size as ytrue_N.

    Returns
    -------
    bce : float
        Binary cross entropy, averaged over all N provided examples
    """
    # Cast labels to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    N = int(ytrue_N.size)
    if N == 0:
        return 0.0

    # Convert binary y values so 0 becomes +1 and 1 becomes -1
    # See HW2 instructions on website for the math
    yflipsign_N = -1 * np.sign(ytrue_N - 0.001)  # dont touch

    # Cast logit scores to float
    scores_N = np.asarray(scores_N, dtype=np.float64)  # dont touch

    flipped_scores_N = yflipsign_N * scores_N

    # Next, stack up two arrays of shape (N,1) to form result of (N,2)
    # First column should be all zero
    # Second column should be flipped_scores_N
    scores_and_zeros_N2 = np.column_stack((np.zeros(N), flipped_scores_N))
    assert (N, 2) == scores_and_zeros_N2.shape

    sigmoid

    # Compute the ultimate BCE score
    # Use scipy_logsumexp from scipy (already imported)
    # Remember: reduction functions let you work along one dim of a 2D array
    # By using the 'axis' keyword argument, like:
    # >>> ans_A = scipy_logsumexp(in_AB, axis=1) # apply logsumexp to each row
    # >>> ans_B = scipy_logsumexp(in_AB, axis=0) # apply logsumexp to each col
    # print(scores_and_zeros_N2)

    bce_score = np.mean(scipy_logsumexp(scores_and_zeros_N2, axis=1) / np.log(2))

    return bce_score
