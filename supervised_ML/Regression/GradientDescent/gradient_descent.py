import numpy as np

class GradientDescent:
    """Core Gradient Descent implementation for Linear Regression (Multiple Features)."""
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.epochs = iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = X.dot(self.weights) + self.bias
            
            # Gradients
            dw = (2/n_samples) * X.T.dot(y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias