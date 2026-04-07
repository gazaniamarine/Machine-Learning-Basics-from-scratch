import numpy as np
import os
import sys

# Add root Classification folder to path for DecisionTree import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DecisionTree.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees, self.max_depth, self.min_samples_split = n_trees, max_depth, min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            t = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            idxs = np.random.choice(X.shape[0], X.shape[0], replace=True)
            t.fit(X[idxs], y[idxs])
            self.trees.append(t)

    def predict(self, X):
        preds = np.swapaxes(np.array([t.predict(X) for t in self.trees]), 0, 1)
        return np.array([np.bincount(p).argmax() for p in preds])
