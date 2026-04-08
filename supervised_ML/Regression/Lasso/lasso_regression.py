import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, lr=0.01, iterations=1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = X.dot(self.weights) + self.bias
            
            # Gradients with L1 penalty: sign(weights)
            dw = (1/n_samples) * (X.T.dot(y_pred - y) + self.alpha * np.sign(self.weights))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias
