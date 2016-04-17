#!/usr/bin/python

from src.experiments import run_experiments_for_five_numbers_of_neurons, select_features
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE

k_best_features = 10


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)
    print("Running experiments with all features")
    run_experiments_for_five_numbers_of_neurons(ds.X, ds.y, algorithm='l-bfgs', max_iter=100,
                                                alpha=1e-5)
    X_selected = select_features(ds.X, ds.y, k_best_features)
    print("Running experiments with selected features")
    run_experiments_for_five_numbers_of_neurons(X_selected, ds.y, algorithm='l-bfgs', max_iter=100,
                                                alpha=1e-5)


if __name__ == '__main__':
    main()
