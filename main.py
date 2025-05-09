from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from time import time

from RandomForest import RandomForest
from TreeLevelParallelRandomForest import TreeLevelParallelRandomForest
from TreeNodeLevelParallelRandomForest import TreeNodeLevelParallelRandomForest

def main():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=5,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42
    )

    n_trees = 50
    max_depth = 50
    min_samples = 5
    n_jobs = 8

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    seeds = np.random.randint(0, 1000, size=n_trees)
    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, min_samples=min_samples)
    start_time = time()
    rf.fit(X_train, y_train, seeds)
    predictions1 = rf.predict(X_test)
    accuracy = np.sum(predictions1 == y_test) / len(y_test)
    end_time = time()
    print(f"Original Random Forest Accuracy: {accuracy:.2f}")
    print(f"Original Random Forest Time: {end_time - start_time:.2f} seconds")

    #rf = TreeLevelParallelRandomForest(n_trees=n_trees, max_depth=max_depth, min_samples=min_samples, n_jobs=n_jobs)
    rf = TreeNodeLevelParallelRandomForest(n_trees=n_trees, max_depth=max_depth, min_samples=min_samples, n_jobs=n_jobs)

    start_time = time()
    rf.fit(X_train, y_train, seeds)
    predictions2 = rf.predict(X_test)
    accuracy = np.sum(predictions2 == y_test) / len(y_test)
    end_time = time()
    print(f"Parallel Random Forest Accuracy: {accuracy:.2f}")
    print(f"Parallel Random Forest Time: {end_time - start_time:.2f} seconds")

    print(predictions1 == predictions2)

if __name__ == "__main__":
    main()

