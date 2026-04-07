import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr, self.epochs = learning_rate, iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-model))
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        model = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-model))
        return [1 if p > 0.5 else 0 for p in y_pred]
