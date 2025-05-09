from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
import pandas as pd

from RandomForest import RandomForest
from ForestLevelParallelRandomForest import ForestLevelParallelRandomForest
from CombinedParallelRandomForest import CombinedParallelRandomForest

class RandomForestBenchmark:
    def __init__(self, n_iterations=10, max_depth=50, min_samples=5,
                 sample_sizes=None, tree_counts=None, thread_counts=None):
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.sample_sizes = sample_sizes or [500, 1000, 2000, 5000]
        self.tree_counts = tree_counts or [10, 20, 50, 100]
        self.thread_counts = thread_counts or [4, 8, 12]
        self.results = []

    def generate_data(self, n_samples):
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=50,
            n_informative=30,
            n_redundant=0,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42
        )

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, seeds):
        start_time = time()
        model.fit(X_train, y_train, seeds)
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        elapsed_time = time() - start_time
        return predictions, accuracy, elapsed_time

    def print_average_results(self, name, accuracies, times):
        print(f"{name:<35} | Accuracy: {np.mean(accuracies):.2%}")
        print(f"{'':<35} | Time:     {np.mean(times):.2f} sec\n")

    def run(self):
        for num_sample in self.sample_sizes:
            X, y = self.generate_data(num_sample)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            for num_tree in self.tree_counts:
                seeds = np.random.randint(0, 1000, size=num_tree)
                for num_thread in self.thread_counts:
                    acc_rf_list, time_rf_list = [], []
                    acc_combined_list, time_combined_list = [], []
                    acc_forest_list, time_forest_list = [], []

                    for _ in range(self.n_iterations):
                        rf = RandomForest(num_tree, self.max_depth, self.min_samples)
                        pred_rf, acc_rf, time_rf = self.evaluate_model(
                            rf, X_train, y_train, X_test, y_test, seeds)
                        acc_rf_list.append(acc_rf)
                        time_rf_list.append(time_rf)

                        combined_rf = CombinedParallelRandomForest(
                            num_tree, self.max_depth, self.min_samples, n_jobs=num_thread)
                        pred_combined, acc_combined, time_combined = self.evaluate_model(
                            combined_rf, X_train, y_train, X_test, y_test, seeds)
                        assert np.array_equal(pred_rf, pred_combined), "Combined RF predictions differ from original!"
                        acc_combined_list.append(acc_combined)
                        time_combined_list.append(time_combined)

                        forest_rf = ForestLevelParallelRandomForest(
                            num_tree, self.max_depth, self.min_samples, n_jobs=num_thread)
                        pred_forest, acc_forest, time_forest = self.evaluate_model(
                            forest_rf, X_train, y_train, X_test, y_test, seeds)
                        assert np.array_equal(pred_rf, pred_forest), "Forest-level RF predictions differ from original!"
                        acc_forest_list.append(acc_forest)
                        time_forest_list.append(time_forest)

                    print(f"\nResults for n_samples={num_sample}, n_trees={num_tree}, n_jobs={num_thread} (avg over {self.n_iterations} runs):")
                    self.print_average_results("Original Random Forest", acc_rf_list, time_rf_list)
                    self.print_average_results("Combined Parallel Random Forest", acc_combined_list, time_combined_list)
                    self.print_average_results("Forest-Level Parallel Random Forest", acc_forest_list, time_forest_list)

                    self.results.extend([
                        {
                            'n_samples': num_sample,
                            'n_trees': num_tree,
                            'n_jobs': num_thread,
                            'model': 'Original Random Forest',
                            'accuracy': np.mean(acc_rf_list),
                            'time_sec': np.mean(time_rf_list)
                        },
                        {
                            'n_samples': num_sample,
                            'n_trees': num_tree,
                            'n_jobs': num_thread,
                            'model': 'Combined Parallel Random Forest',
                            'accuracy': np.mean(acc_combined_list),
                            'time_sec': np.mean(time_combined_list)
                        },
                        {
                            'n_samples': num_sample,
                            'n_trees': num_tree,
                            'n_jobs': num_thread,
                            'model': 'Forest-Level Parallel Random Forest',
                            'accuracy': np.mean(acc_forest_list),
                            'time_sec': np.mean(time_forest_list)
                        }
                    ])

    def save_results(self, filename="random_forest_benchmark.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nAll results saved to '{filename}'")


if __name__ == "__main__":
    benchmark = RandomForestBenchmark()
    benchmark.run()
    benchmark.save_results()