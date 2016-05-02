#!/usr/bin/python

from src.experiments import run_experiment
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE
import matplotlib.pyplot as plt


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    print("Ranking (descending)", ds.create_features_ranking(use_names=True))
    results = {}
    for n_features in range(1, ds.number_of_features, 1):
        results[n_features] = run_experiment(ds.X, ds.y, algorithm='sgd', max_iter=1000,
                                             alpha=1e-6, learning_rate='constant',
                                             n_features=n_features, hidden_layer_size=n_features)
    y = []
    for n_features, score in sorted(results.items()):
        print("Score for {} features = {}".format(n_features, score))
        y.append(score)

    x = range(1, ds.number_of_features, 1)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main()
