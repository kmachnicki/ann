#!/usr/bin/python

from src.experiments import run_experiment
from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE


def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)
    for layer_size in range(50, 550, 100):
        print("Running experiment for layer_size =", layer_size)
        run_experiment(ds.X, ds.y, algorithm='l-bfgs', max_iter=100, alpha=1e-5, hidden_layer_sizes=layer_size)

if __name__ == '__main__':
    main()
