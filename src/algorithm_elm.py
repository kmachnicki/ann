#!/usr/bin/python

from modules.elm import ELMClassifier
from time import time
import numpy as np

from src.dataset import DataSet
from src.consts import INPUT_DATA_FILE

from pylab import mean, std, scatter, show
from sklearn.model_selection import train_test_split


def res_dist(x, y, e, n_runs=100, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=random_state)

    test_res = []
    train_res = []
    start_time = time()

    for i in range(n_runs):
        print("Run: %d" % i)
        e.fit(x_train, y_train)
        train_res.append(e.score(x_train, y_train))
        test_res.append(e.score(x_test, y_test))

    print("\nTime: %.3f secs" % (time() - start_time))

    print("Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f"
          % (min(test_res), mean(test_res), max(test_res), std(test_res)))
    print("Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f"
          % (min(train_res), mean(train_res), max(train_res), std(train_res)))

    return train_res, test_res

def run_elm(X, y):
    '''
    Extreme learning machine algorithm for the single layer feedforward network (SLFN)
    '''
    elmc = ELMClassifier(n_hidden=500, activation_func='multiquadric')
    tr, ts = res_dist(X, y, elmc, n_runs=100, random_state=0)
    scatter(tr, ts, alpha=0.1, marker='D', c='r')
    show()

def main():
    ds = DataSet()
    with open(INPUT_DATA_FILE, "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

    run_elm(ds.X, ds.y)


if __name__ == '__main__':
    main()
