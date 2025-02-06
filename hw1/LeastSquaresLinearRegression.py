"""
Test Cases
----------
# Same test as test_on_toy_data() defined below

>>> prng = np.random.RandomState(0)
>>> N = 100

>>> true_w_F = np.asarray([1.1, -2.2, 3.3])
>>> true_b = 0.0
>>> x_NF = prng.randn(N, 3)
>>> y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)

>>> linear_regr = LeastSquaresLinearRegressor()
>>> linear_regr.fit(x_NF, y_N)

>>> yhat_N = linear_regr.predict(x_NF)
>>> np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})
>>> print(linear_regr.w_F)
[ 1.099 -2.202  3.301]
>>> print(np.asarray([linear_regr.b]))
[-0.005]
"""

import numpy as np

# No other imports allowed!


class LeastSquaresLinearRegressor(object):
    """A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    """

    def __init__(self):
        """Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        """
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        """Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing.

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:

        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        """
        N, F = x_NF.shape
        G = F + 1

        xtilde_NG = np.hstack((x_NF, np.ones((N, 1))))
        assert xtilde_NG.shape == (N, G)

        xtildeT_GN = xtilde_NG.T
        assert xtildeT_GN.shape == (G, N)

        xTx_GG = xtildeT_GN @ xtilde_NG
        assert xTx_GG.shape == (G, G)

        RHS_G1 = (xtildeT_GN @ y_N)[:, np.newaxis]
        assert RHS_G1.shape == (G, 1)

        LHS_GG = xTx_GG

        theta_G1 = np.linalg.solve(LHS_GG, RHS_G1)
        assert theta_G1.shape == (G, 1)

        self.w_F = theta_G1[:F].flatten()
        assert self.w_F.shape == (F,)
        # should not be a scalar
        # assert not isinstance(self.w_F, np.float64)

        self.b = theta_G1[-1][0]
        # should be a scalar
        # assert isinstance(self.b, np.float64)

    def predict(self, x_MF):
        """Make predictions given input features for M examples

        Args
        ----
        x_MF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        """
        M, F = x_MF.shape
        # assert self.w_F.shape == (F,)

        # for x_F in x_MF:
        #     assert x_F.shape == (F,)
        # (M, F) (F, 1) => (M, 1)
        # Adds the bias to the dot product of each row with the weights
        yhat_M = self.b + (x_MF @ self.w_F)

        # assert yhat_M.shape == (M,)
        return yhat_M


def test_on_toy_data(N=100):
    """
    Simple test case with toy dataset with N=100 examples
    created via a known linear regression model plus small noise.

    The test verifies that our LR can recover true w and b parameter values.
    """
    prng = np.random.RandomState(0)
    N = 100

    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.matmul(x_NF, true_w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)

    yhat_N = linear_regr.predict(x_NF)
    np.set_printoptions(precision=3, formatter={"float": lambda x: "% .3f" % x})
    print(linear_regr.w_F)

    print(np.asarray([linear_regr.b]))
