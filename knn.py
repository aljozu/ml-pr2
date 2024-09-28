import numpy as np
from scipy.spatial import KDTree
from collections import Counter
from joblib import Parallel, delayed

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        # Build a KDTree for efficient neighbor lookup
        self.tree = KDTree(X_train)

    def _predict_single(self, x):
        # Query the KDTree for the k-nearest neighbors
        distances, k_indices = self.tree.query(x, k=self.n_neighbors)

        # Retrieve the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        # Parallelize prediction over multiple samples
        y_pred = Parallel(n_jobs=-1, prefer="threads")(delayed(self._predict_single)(x) for x in X_test)
        return np.array(y_pred)

    def _predict_single(self, x):
        # Query the KDTree for the k-nearest neighbors
        distances, k_indices = self.tree.query(x, k=self.n_neighbors)

        # Retrieve the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
