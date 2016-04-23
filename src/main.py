#!/usr/bin/python

from src.experiments import run_experiments_for_five_numbers_of_neurons
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)
    print("Ranking", ds.create_features_ranking(use_names=False))
    run_experiments_for_five_numbers_of_neurons(ds.X, ds.y, algorithm='sgd', max_iter=1000,
                                                alpha=1e-6, learning_rate='adaptive', n_features=10)

if __name__ == '__main__':
    main()
