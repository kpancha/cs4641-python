import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        error = pred - label
        return np.sqrt(np.mean(np.square(error)))

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        N = x.shape[0]
        feat = np.ones((N, degree + 1))
        feat[:, 1] = x
        for i in range(N):
            for j in range(1, degree + 1):
                feat[i,j] = np.power(feat[i, 1], j)
        return feat

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        prediction = np.dot(xtest, weight)
        return prediction

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        xty = np.matmul(xtrain.T, ytrain)
        xtx_inv = np.linalg.inv(np.matmul(xtrain.T, xtrain))
        weight = np.matmul(xtx_inv, xty)
        return weight

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        for i in range(epochs):
            for j in range(N):
                xw = np.matmul(xtrain[j:j+1], weight)
                grad = np.matmul(xtrain[j:j+1].T, ytrain[j:j+1] - xw)
                weight += learning_rate * grad
        return weight

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        for i in range(epochs):
            xw = np.matmul(xtrain, weight)
            weight += learning_rate * np.dot(xtrain.T, ytrain - xw) / N
        return weight

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        _, D = xtrain.shape
        I = np.identity(D)
        I[0, :] = 0
        xtx = np.dot(xtrain.T, xtrain)
        xty = np.dot(xtrain.T, ytrain)
        weight = np.dot(np.linalg.inv(xtx + c_lambda*I), xty)
        return weight

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        weight = np.zeros((D,1))
        c_lambda *= np.ones(weight.shape)
        c_lambda[0,0] = 0.0
        for i in range(epochs):
            xw = np.dot(xtrain, weight)
            loss = np.matmul(xtrain.T, ytrain - xw) + c_lambda * weight
            loss /= N
            weight += learning_rate * loss
        return weight

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        N = X.shape[0]
        rmses = []
        points_per_fold = np.int(N / kfold)
        for i in range(kfold):
            j = i+1
            
            xtest = X[i*points_per_fold:j*points_per_fold, :]
            ytest = y[i*points_per_fold:j*points_per_fold, :]
            
            x1 = X[:i*points_per_fold]
            x2 = X[j*points_per_fold:]
            y1 = y[:i*points_per_fold, :]
            y2 = y[j*points_per_fold:, :]
            
            xtrain = np.concatenate((x1,x2), axis=0)
            ytrain = np.concatenate((y1,y2), axis=0)
            
            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
            predict_ytest = self.predict(xtest, weight)
            rmses.append(self.rmse(predict_ytest, ytest))
            
        meanErrors = np.mean(rmses)
        return meanErrors
