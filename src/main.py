#!/usr/bin/python

from src.experiments import run_experiments_for_five_numbers_of_neurons
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)
    run_experiments_for_five_numbers_of_neurons(ds.X, ds.y, algorithm='l-bfgs', max_iter=150,
                                                alpha=1e-5, n_features=2, min_layer_size=100)

if __name__ == '__main__':
    main()
