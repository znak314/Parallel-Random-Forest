import numpy as np
from DecisionTree import DecisionTree
from multiprocessing import Pool


def build_tree_for_forest(X, y, seed, max_depth, min_samples):
    np.random.seed(seed)
    indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
    tree = DecisionTree(max_depth=max_depth, min_samples=min_samples, parallel=True)
    tree.fit(X[indices], y[indices])
    return tree

def predict_tree(tree, X):
    return tree.predict(X)


class CombinedParallelRandomForest:
    def __init__(self, n_trees=3, max_depth=50, min_samples=5, n_jobs=1, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.random_state = np.random.RandomState(random_state)
        self.trees = []

    def fit(self, X, y, seeds):
        args = [(X, y, seed, self.max_depth, self.min_samples) for seed in seeds]
        with Pool(processes=max(1, self.n_jobs // 2)) as pool:
            self.trees = pool.starmap(build_tree_for_forest, args)

    def predict(self, X):
        with Pool(processes=self.n_jobs) as pool:
            predictions = pool.starmap(predict_tree, [(tree, X) for tree in self.trees])

        predictions = np.array(predictions).T
        return [self.trees[0].mode(predictions[i, :]) for i in range(X.shape[0])]
