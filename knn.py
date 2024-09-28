import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x):
        # Calculate distances to all training points
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # Retrieve the classes of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")




