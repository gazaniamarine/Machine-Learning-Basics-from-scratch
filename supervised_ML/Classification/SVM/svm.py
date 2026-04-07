import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.lr, self.lambda_p, self.epochs = learning_rate, lambda_param, iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                condition = y[i] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_p * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_p * self.weights - np.dot(x_i, y[i]))
                    self.bias -= self.lr * y[i]

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
