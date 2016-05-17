import numpy as np
import matplotlib.pyplot as plt

from src.consts import HIDDEN_LAYER_SIZES


def generate_plots(results, n_features):
    x = range(1, n_features, 1)
    for layer_size in HIDDEN_LAYER_SIZES:
        draw(x, layer_size, results, n_features, "bp", "scores", "Score")
        draw(x, layer_size, results, n_features, "bp", "fit_times", "Fit time")
        draw(x, layer_size, results, n_features, "bp", "score_times", "Score time")

        draw(x, layer_size, results, n_features, "elm", "scores", "Score")
        draw(x, layer_size, results, n_features, "elm", "fit_times", "Fit time")
        draw(x, layer_size, results, n_features, "elm", "score_times", "Score time")


def draw(x, layer_size, results, n_features, alg_type, sample_type, ylabel):
    y = []
    e = []
    for n_features, result in sorted(results[layer_size].items()):
        y.append(np.mean(result[alg_type][sample_type]))
        e.append(np.std(result[alg_type][sample_type]))
    plot(x, y, e, layer_size, n_features, alg_type, sample_type, ylabel)


def plot(x, y, e, layer_size, n_features, alg_type, sample_type, ylabel):
    plt.figure(layer_size)
    plt.errorbar(x, y, yerr=e, fmt="bo--", ecolor="b", linewidth=1.0)

    if alg_type == "bp":
        plt.title("Back propagation, hidden layer size: " + str(layer_size))
    else:
        plt.title("Extreme learning machine, hidden layer size: " + str(layer_size))

    if sample_type == "score":
        plt.ylim(0.0, 1.0)
        plt.yticks(np.arange(0.1, 1.0, 0.1))

    plt.xlim(1.0, n_features - 1)
    plt.xticks(x, rotation="vertical")
    plt.ylabel(ylabel)
    plt.xlabel("Number of features")
    plt.grid()
    plt.show()
    #plt.savefig(alg_type + "_" + sample_type + "_" + str(layer_size) + ".png")
