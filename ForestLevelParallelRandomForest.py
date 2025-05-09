import numpy as np
from joblib import Parallel, delayed
from DecisionTree import DecisionTree


class ForestLevelParallelRandomForest:
    def __init__(self, n_trees=3, max_depth=50, min_samples=5, n_jobs=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_jobs = n_jobs 

        self.trees = []
        self.n_samples = None

    def fit(self, X, y, seeds):
        self.n_samples = X.shape[0]
        
        def train_tree(i):
            np.random.seed(seeds[i])
            samples = np.random.choice(self.n_samples, self.n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X[samples, :], y[samples])
            return tree
        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(train_tree)(i) for i in range(self.n_trees)
        )
        
    def predict(self, X):
        def predict_tree(tree, X):
            return tree.predict(X)
        
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_tree)(tree, X) for tree in self.trees
        )

        predictions = np.array(predictions).T 
        return [self.trees[0].mode(predictions[i, :]) for i in range(X.shape[0])]
    
