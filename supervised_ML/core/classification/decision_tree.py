import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right, self.value = left, right, value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split, self.max_depth = min_samples_split, max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y)

    def _grow(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split):
            return Node(value=np.bincount(y).argmax())

        best_feat, best_thresh = self._best_split(X, y, np.random.choice(n_feats, n_feats, replace=False))
        left_idx = np.where(X[:, best_feat] <= best_thresh)[0]
        right_idx = np.where(X[:, best_feat] > best_thresh)[0]
        
        left = self._grow(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain, split_idx, split_thresh = -1, None, None
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            for thr in np.unique(X_col):
                gain = self._gain(y, X_col, thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thr
        return split_idx, split_thresh

    def _gain(self, y, X_col, thr):
        p_ent = self._entropy(y)
        left_idx, right_idx = np.where(X_col <= thr)[0], np.where(X_col > thr)[0]
        if len(left_idx) == 0 or len(right_idx) == 0: return 0
        
        n, n_l, n_r = len(y), len(left_idx), len(right_idx)
        c_ent = (n_l/n) * self._entropy(y[left_idx]) + (n_r/n) * self._entropy(y[right_idx])
        return p_ent - c_ent

    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None: return node.value
        return self._traverse(x, node.left) if x[node.feature] <= node.threshold else self._traverse(x, node.right)
