#!/usr/bin/python

from src.algorithms import run_experiment
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    print("Ranking (descending)", ds.create_features_ranking(use_names=True))

    experiment_results = {}
    hidden_layer_sizes = [5, 50]#[100, 250, 500, 750, 1000]
    final_counter = Counter()

    for layer_size in hidden_layer_sizes:
        experiment_results[layer_size] = {}
        for n_features in range(1, ds.number_of_features, 1):
            experiment_results[layer_size][n_features] = \
                run_experiment(ds.X, ds.y, hidden_layer_size=layer_size, n_features=n_features, algorithm='sgd',
                               max_iter=1000, alpha=1e-6, learning_rate='constant', kfold=10,
                               activation_func='multiquadric')
            print("\n\nLayer size: {}, number of features: {}".format(layer_size, n_features))
            print("BP: score: {:.3f}, fit time: {:.8f} secs, score time: {:.8f} secs"
                  .format(experiment_results[layer_size][n_features].bp_results.score,
                          experiment_results[layer_size][n_features].bp_results.fit_time,
                          experiment_results[layer_size][n_features].bp_results.score_time))
            print("ELM: score: {:.3f}, fit time: {:.8f} secs, score time: {:.8f} secs"
                  .format(experiment_results[layer_size][n_features].elm_results.score,
                          experiment_results[layer_size][n_features].elm_results.fit_time,
                          experiment_results[layer_size][n_features].elm_results.score_time))

            final_counter.update(experiment_results[layer_size][n_features].counter)

    print("Num of times features were selected: {}".format(final_counter))

    generate_plots(experiment_results, "bp", ds.number_of_features, hidden_layer_sizes)
    generate_plots(experiment_results, "elm", ds.number_of_features, hidden_layer_sizes)


def generate_plots(results, prefix, number_of_features, hidden_layer_sizes):
    x = range(1, number_of_features, 1)
    for layer_size in hidden_layer_sizes:
        y = []
        for n_features, score in sorted(results[layer_size].items()):
            y.append(score)
        plt.figure(layer_size)
        plt.plot(x, y)
        if prefix == "bp":
            plt.title("Back propagation, hidden layer size: " + str(layer_size))
        else:
            plt.title("Extreme learning machine, hidden layer size: " + str(layer_size))
        plt.xlim(1.0, number_of_features)
        plt.ylim(0.0, 1.0)
        plt.xticks(x, rotation="vertical")
        plt.yticks(np.arange(0.1, 1.0, 0.1))
        plt.ylabel("Score")
        plt.xlabel("Number of features")
        plt.grid()
        plt.savefig(prefix + "_" + str(layer_size) + ".png")

if __name__ == '__main__':
    main()
