import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion 
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Check stopping criteria 
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return Node(value=self._most_common_label(y))  # Create leaf node

        # Find the best split based on the chosen criterion
        feature, threshold = self._best_split(X, y)
        
        # Split data
        left_idxs, right_idxs = self._split(X[:, feature], threshold)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_impurity = float('inf')  # Lower impurity is better
        
        # Loop over all features and thresholds
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature], threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue  # Skip if no split is possible
                
                # Calculate the weighted impurity of this split
                impurity = self._weighted_impurity(y, left_idxs, right_idxs)
                
                # Update if a better (lower) impurity is found
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _weighted_impurity(self, y, left_idxs, right_idxs):
        # Get sizes of left and right splits
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        # Compute the impurity for the left and right splits
        if n_left == 0 or n_right == 0:
            return float('inf')

        # Calculate the weighted average of the impurities
        left_impurity = self._impurity(y[left_idxs])
        right_impurity = self._impurity(y[right_idxs])
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        return weighted_impurity

    def _impurity(self, y):
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        elif self.criterion == "misclassification":
            return self._misclassification_error(y)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        class_probs = np.bincount(y) / m
        return 1 - np.sum(class_probs ** 2)

    def _entropy(self, y):
        m = len(y)
        if m == 0:
            return 0
        class_probs = np.bincount(y) / m
        return -np.sum([p * np.log2(p) for p in class_probs if p > 0])

    def _misclassification_error(self, y):
        m = len(y)
        if m == 0:
            return 0
        class_probs = np.bincount(y) / m
        return 1 - np.max(class_probs)

    def _split(self, feature_column, threshold):
        left_idxs = np.argwhere(feature_column <= threshold).flatten()
        right_idxs = np.argwhere(feature_column > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()