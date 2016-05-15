#!/usr/bin/python

from src.algorithm_bp import run_bp
from src.algorithm_elm import run_elm
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE
import matplotlib.pyplot as plt
from collections import Counter
from time import time
import numpy as np


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    print("Ranking (descending)", ds.create_features_ranking(use_names=True))
    results_bp = {}
    results_elm = {}
    hidden_layer_sizes = [5, 50]#[100, 250, 500, 750, 1000]
    final_counter_bp = Counter()
    for layer_size in hidden_layer_sizes:
        results_bp[layer_size] = {}
        results_elm[layer_size] = {}
        for n_features in range(1, ds.number_of_features, 1):
            start_time = time()
            results_bp[layer_size][n_features], counter_bp = \
                run_bp(ds.X, ds.y, algorithm='sgd', max_iter=1000,
                       alpha=1e-6, learning_rate='constant',
                       n_features=n_features, hidden_layer_size=layer_size)
            print("BP: Layer size: {}, features: {}, score: {:.3f}, time: {:.5f} secs"
                  .format(layer_size, n_features, results_bp[layer_size][n_features], time() - start_time))
            final_counter_bp.update(counter_bp)

            '''
            results_elm[layer_size][n_features] = run_elm(ds.X, ds.y)
            print("ELM: Layer size: {}, features: {}, score: {:.3f}, time: {:.5f} secs"
                  .format(layer_size, n_features, results_elm[layer_size][n_features], time() - start_time))
            '''

    print("Num of times features were selected: {}".format(final_counter_bp))

    generate_plots(results_bp, "bp", ds.number_of_features, hidden_layer_sizes)
    #generate_plots(results_elm, "elm", ds.number_of_features, hidden_layer_sizes)


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
