#!/usr/bin/python

from src.experiments import run_experiments_for_five_numbers_of_neurons
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)
    print("Ranking (descending)", ds.create_features_ranking(use_names=True))
    results = {}
    for n_features in range(2, ds.number_of_features, 1):
        results[n_features] =  run_experiments_for_five_numbers_of_neurons(ds.X, ds.y, algorithm='sgd', max_iter=10000,
                                                alpha=1e-6, learning_rate='adaptive', n_features=n_features)
    for n_features, score in sorted(results.items()):
        print("Score (mean, std, layer_size) for {} features = {}, {}".format(n_features, score, score[0] - score[1]))

if __name__ == '__main__':
    main()
