import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        # Add bias column (X_b = [1, X])
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # theta = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias, self.weights = theta[0], theta[1:]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias
