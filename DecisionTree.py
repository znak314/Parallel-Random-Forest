import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, value = None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value

  def is_leaf_node(self):
    return self.value is not None
    

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=10, parallel=False):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.parallel = parallel
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
    
    def parallel_best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1
        
        n_columns_to_select = int(np.sqrt(X.shape[1]))
        indices = np.random.choice(X.shape[1], size=n_columns_to_select, replace=False)

        def process_column(i):
            thresholds = np.unique(X[:, i])
            local_best_gain = -1
            local_best_threshold = None
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > local_best_gain:
                    local_best_gain = gain
                    local_best_threshold = threshold
            return i, local_best_threshold, local_best_gain
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(process_column, indices))

        for i, local_best_threshold, local_best_gain in results:
            if local_best_gain > best_gain:
                best_gain = local_best_gain
                best_feature = i
                best_threshold = local_best_threshold

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
        
        if self.parallel:
            best_feature, best_threshold = self.parallel_best_split(X, y)
        else:
            best_feature, best_threshold = self.best_split(X, y)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()
        
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.mode(y))
        
        left = self.build_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self.build_tree(X[right_indexes, :], y[right_indexes], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def traverse_tree(self, x, tree):
        if tree.is_leaf_node():
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return self.traverse_tree(x, tree.left)
        return self.traverse_tree(x, tree.right)