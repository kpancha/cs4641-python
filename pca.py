import numpy as np


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X):
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        col_means = np.mean(X, axis=0)
        X -= col_means
        U, S, V = np.linalg.svd(X, full_matrices=False)
        self.U = U
        self.S = S
        self.V = V

    def transform(self, data, K=2):
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        X_new = self.U[:, :K]
        X_new *= self.S[:K]
        return X_new

    def transform_rv(self, data, retained_variance=0.99):
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """

        S = np.square(self.S)
        total_var = np.sum(S)
        S = S[np.argsort(S)[::-1]]
        cumsum = np.cumsum(S)
        k = np.argwhere(cumsum / total_var >= retained_variance)[0][0] + 1
        return self.transform(data,k)

    def get_V(self):
        """ Getter function for value of V """

        return self.V
