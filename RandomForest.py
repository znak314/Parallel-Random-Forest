import numpy as np
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=3, max_depth=50, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_samples = None
        self.trees = []

    def fit(self, X, y, seeds):
        self.n_samples = X.shape[0]
        for i in range(self.n_trees):
            np.random.seed(seeds[i])
            samples = np.random.choice(self.n_samples, self.n_samples, replace = True)
            tree = DecisionTree(max_depth = self.max_depth, min_samples=self.min_samples)
            tree.fit(X[samples,:], y[samples])
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = []
        for i in range(self.n_trees):
            predictions.append(self.trees[i].predict(X))
        predictions = [self.trees[0].mode(np.array(predictions)[:,i]) for i in range(X.shape[0])]
        return predictions
