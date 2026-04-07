import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X, y=None):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Randomly initialize centroids by picking distinct data points
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idx]
        
        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros((self.n_clusters, X.shape[1]))
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = X[np.random.choice(X.shape[0])]
            
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.centroids = new_centroids
                self.labels_ = labels
                break
                
            self.centroids = new_centroids
            self.labels_ = labels
            
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
        return self
        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
        
    def _compute_distances(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, c in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - c, axis=1)
        return distances

    def _compute_inertia(self, X, labels, centroids):
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia
