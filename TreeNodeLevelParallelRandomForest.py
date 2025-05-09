import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Node class definition remains unchanged
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=10, n_jobs=1):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_jobs = n_jobs  # Add n_jobs parameter
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.tree) for x in X])

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def mode(self, y):
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]

    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1

        n_columns_to_select = int(np.sqrt(X.shape[1]))
        indices = np.random.choice(X.shape[1], size=n_columns_to_select, replace=False)
        for i in indices:
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        return best_feature, best_threshold

    def information_gain(self, X_column, y, threshold):
        if len(np.unique(y)) == 1:
            return 0

        parent_entropy = self.entropy(y)

        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()

        left_entropy, left_samples = self.entropy(y[left_indexes]), len(left_indexes)
        right_entropy, right_samples = self.entropy(y[right_indexes]), len(right_indexes)

        num_samples = len(y)
        child_entropy = (left_samples / num_samples) * left_entropy + (right_samples / num_samples) * right_entropy
        return parent_entropy - child_entropy

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_samples <= self.min_samples or depth >= self.max_depth or n_labels == 1:
            return Node(value=self.mode(y))

        best_feature, best_threshold = self.best_split(X, y)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.mode(y))

        # Add parallelization for building left and right subtrees
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self.build_tree, X[left_indexes, :], y[left_indexes], depth + 1),
                executor.submit(self.build_tree, X[right_indexes, :], y[right_indexes], depth + 1)
            ]
            left = futures[0].result()
            right = futures[1].result()

        return Node(best_feature, best_threshold, left, right)

    def traverse_tree(self, x, tree):
        if tree.is_leaf_node():
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return self.traverse_tree(x, tree.left)
        return self.traverse_tree(x, tree.right)

# RandomForest class with parallel tree building
class TreeNodeLevelParallelRandomForest:
    def __init__(self, n_trees=3, max_depth=50, min_samples=5, n_jobs=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_samples = None
        self.n_jobs = n_jobs 
        self.trees = []

    def fit(self, X, y, seeds):
        self.n_samples = X.shape[0]
        
        with ProcessPoolExecutor(max_workers = self.n_jobs // 2) as executor:
            futures = [executor.submit(self._build_tree, X, y, seeds[i]) for i in range(self.n_trees)]
            for future in futures:
                self.trees.append(future.result())
        
    def _build_tree(self, X, y, seed):
        np.random.seed(seed)
        samples = np.random.choice(self.n_samples, self.n_samples, replace=True)
        tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
        tree.fit(X[samples, :], y[samples])
        return tree

    def predict(self, X):
        predictions = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self._predict_single_tree, X, i) for i in range(self.n_trees)]
            for future in futures:
                predictions.append(future.result())
        
        predictions = np.array(predictions).T
        return [self.mode(pred) for pred in predictions]

    def _predict_single_tree(self, X, tree_index):
        return self.trees[tree_index].predict(X)

    def mode(self, y):
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]

