import numpy as np
import matplotlib.pyplot as plt

from src.consts import HIDDEN_LAYER_SIZES


def generate_plots(results, number_of_features):
    x = range(1, number_of_features, 1)
    for layer_size in HIDDEN_LAYER_SIZES:
        draw_bp_score(x, layer_size, results, number_of_features)
        draw_bp_fit_time(x, layer_size, results, number_of_features)
        draw_bp_score_time(x, layer_size, results, number_of_features)

        draw_elm_score(x, layer_size, results, number_of_features)
        draw_elm_fit_time(x, layer_size, results, number_of_features)
        draw_elm_score_time(x, layer_size, results, number_of_features)


def draw_bp_score(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.bp_samples.scores))
    plot(x, y, layer_size, number_of_features, "bp_score", "Score",
         "Back propagation, hidden layer size: " + str(layer_size))


def draw_bp_fit_time(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.bp_samples.fit_times))
    plot(x, y, layer_size, number_of_features, "bp_fit_time", "Fit time",
         "Back propagation, hidden layer size: " + str(layer_size))


def draw_bp_score_time(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.bp_samples.score_times))
    plot(x, y, layer_size, number_of_features, "bp_score_time", "Score time",
         "Back propagation, hidden layer size: " + str(layer_size))


def draw_elm_score(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.elm_samples.scores))
    plot(x, y, layer_size, number_of_features, "elm_score", "Score",
         "Extreme learning machine, hidden layer size: " + str(layer_size))


def draw_elm_fit_time(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.elm_samples.fit_times))
    plot(x, y, layer_size, number_of_features, "elm_fit_time", "Fit time",
         "Extreme learning machine, hidden layer size: " + str(layer_size))


def draw_elm_score_time(x, layer_size, results, number_of_features):
    y = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result.elm_samples.score_times))
    plot(x, y, layer_size, number_of_features, "elm_score_time", "Score time",
         "Extreme learning machine, hidden layer size: " + str(layer_size))


def plot(x, y, layer_size, number_of_features, prefix, ylabel, title):
    plt.figure(layer_size)
    plt.plot(x, y)
    plt.title(title)
    plt.xlim(1.0, number_of_features - 1)
    plt.ylim(0.0, 1.0)
    plt.xticks(x, rotation="vertical")
    plt.yticks(np.arange(0.1, 1.0, 0.1))
    plt.ylabel(ylabel)
    plt.xlabel("Number of features")
    plt.grid()
    plt.show()
    #plt.savefig(prefix + "_" + str(layer_size) + ".png")
