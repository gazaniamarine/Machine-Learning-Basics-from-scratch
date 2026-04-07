import numpy as np

class GradientDescent:
    """Core Gradient Descent implementation for Linear Regression."""
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.epochs = iterations
        self.m, self.b = 0, 0

    def fit(self, x, y):
        n = len(x)
        for i in range(self.epochs):
            y_pred = self.m * x + self.b
            
            # Gradients
            md = -(2/n) * sum(x * (y - y_pred))
            bd = -(2/n) * sum(y - y_pred)
            
            # Update
            self.m -= self.lr * md
            self.b -= self.lr * bd

    def predict(self, x):
        return self.m * x + self.b