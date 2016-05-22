from os import path, makedirs

import numpy as np
import matplotlib.pyplot as plt

from src.consts import HIDDEN_LAYER_SIZES, OUTPUT_IMAGES_DIR
from src.helpers import GraphValuesWrapper


def ensure_dir(dir_path):
    if not path.exists(dir_path):
        makedirs(dir_path)


def generate_plots(results, n_features, col_names, counter):
    ensure_dir(OUTPUT_IMAGES_DIR)

    draw_counter(col_names, counter)
    draw_conf_matrix(results)

    x = range(1, n_features, 1)
    for layer_size in HIDDEN_LAYER_SIZES:
        draw_graphs(x, layer_size, results, n_features)

    plt.show()


def draw_conf_matrix(results):
    plt.figure(2)
    for n_features, result in sorted(results[HIDDEN_LAYER_SIZES[0]].items()):
        cm = result["bp"]["conf_matrices"]
        plt.imshow(cm[0], interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(cm[0]))
        plt.xticks(tick_marks, ["G2", "G3"], rotation=45)
        plt.yticks(tick_marks, ["G2", "G3"])
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(path.join(OUTPUT_IMAGES_DIR, "conf_matrix.pdf"))
        return


def draw_counter(col_names, counter):
    plt.figure(1)
    labels, values = zip(*counter.items())

    indexes = np.arange(len(labels))
    width = 1.0

    plt.bar(indexes, values, width=width)
    plt.xlim(0.0, len(indexes))
    plt.xticks(indexes, col_names, rotation=45)
    plt.savefig(path.join(OUTPUT_IMAGES_DIR, "features.pdf"))


def draw_graphs(x, layer_size, results, n_features):
    score_values = GraphValuesWrapper()
    fit_time_values = GraphValuesWrapper()
    score_time_values = GraphValuesWrapper()

    for n_features, result in sorted(results[layer_size].items()):
        score_values.add_value(result["bp"]["scores"], result["elm"]["scores"])
        fit_time_values.add_value(result["bp"]["fit_times"], result["elm"]["fit_times"])
        score_time_values.add_value(result["bp"]["score_times"], result["elm"]["score_times"])

    plt.figure(layer_size)
    plt.title("BP & ELM, hidden layer size: " + str(layer_size))
    plt.xlabel("Number of features")
    plot(x, score_values.values(), n_features, "score", "Score", 311)
    plot(x, fit_time_values.values(), n_features, "fit_time", "Fit time", 312)
    plot(x, score_time_values.values(), n_features, "score_time", "Score time", 313)
    plt.savefig(path.join(OUTPUT_IMAGES_DIR, "bp_elm_" + str(layer_size) + ".pdf"))


def plot(x, values, n_features, sample_type, ylabel, subplot_pos):
    plt.subplot(subplot_pos)
    plt.errorbar(x, values.y_bp, yerr=values.e_bp, fmt="bo--", ecolor="b", linewidth=1.0, label="BP")
    plt.errorbar(x, values.y_elm, yerr=values.e_elm, fmt="go--", ecolor="g", linewidth=1.0, label="ELM")

    if sample_type == "score":
        plt.ylim(0.0, 1.0)
        plt.yticks(np.arange(0.1, 1.0, 0.1))

    plt.xlim(1.0, n_features - 1)
    plt.xticks(x, rotation="vertical")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc="best", prop={"size": 11})
