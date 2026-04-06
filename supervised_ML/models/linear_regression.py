import numpy as np

class LinearRegression:
    """Mathematical implementation of Linear Regression (Multivariate) using the Normal Equation."""
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fits the linear regression model. (X^T * X)^(-1) * X^T * y"""
        X_array, y_array = np.array(X), np.array(y)
        # Ensure 2D for X
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # Add bias column (X_b = [1, X])
        X_b = np.c_[np.ones((X_array.shape[0], 1)), X_array]
        
        # Solving using the Pseudo-Inverse for numerical stability
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_array)
        
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Predicts using the learned weights and bias."""
        X_array = np.array(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        return X_array.dot(self.weights) + self.bias
