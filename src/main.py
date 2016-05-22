#!/usr/bin/python

from collections import Counter
import numpy as np

from src.algorithms import run_experiment
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE, HIDDEN_LAYER_SIZES
from src.grapher import generate_plots


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    print("Ranking (descending)", ds.create_features_ranking(use_names=True))

    experiment_results = {}
    final_counter = Counter()

    for layer_size in HIDDEN_LAYER_SIZES:
        experiment_results[layer_size] = {}
        for n_features in range(1, ds.number_of_features, 1):
            result = run_experiment(ds.X, ds.y, hidden_layer_size=layer_size, n_features=n_features)
            experiment_results[layer_size][n_features] = result
            final_counter.update(result.counter)
            print_result(result, layer_size, n_features)

    print("\nNum of times features were selected: {}".format(final_counter))

    generate_plots(experiment_results, ds.number_of_features, ds.col_names, final_counter)


def print_result(result, layer_size, n_features):
    print("\n\nLayer size: {}, number of features: {}".format(layer_size, n_features))

    print("BP:  avg score: {:.3f}, min score: {:.3f}, max score: {:.3f}, SD: {:.3f}"
          .format(np.mean(result.bp_samples.scores),
                  np.min(result.bp_samples.scores),
                  np.max(result.bp_samples.scores),
                  np.std(result.bp_samples.scores)))
    print("     avg fit time: {:.6f} secs, avg score time: {:.6f} secs"
          .format(np.mean(result.bp_samples.fit_times),
                  np.mean(result.bp_samples.score_times)))

    print("ELM: avg score: {:.3f}, min score: {:.3f}, max score: {:.3f}, SD: {:.3f}"
          .format(np.mean(result.elm_samples.scores),
                  np.min(result.elm_samples.scores),
                  np.max(result.elm_samples.scores),
                  np.std(result.elm_samples.scores)))
    print("     avg fit time: {:.6f} secs, avg score time: {:.6f} secs"
          .format(np.mean(result.elm_samples.fit_times),
                  np.mean(result.elm_samples.score_times)))


if __name__ == '__main__':
    main()
