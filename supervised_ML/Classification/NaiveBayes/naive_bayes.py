import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.priors = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.priors = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])
            feature_counts = X_c.sum(axis=0)
            total_count = feature_counts.sum()
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + (self.alpha * n_features)
            self.feature_log_prob_[idx, :] = np.log(smoothed_counts / smoothed_total)

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = prior + np.sum(self.feature_log_prob_[idx] * x)
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
