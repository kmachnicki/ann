#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def run_experiment(X_train, y, algorithm='l-bfgs', max_iter=100, alpha=1e-6, hidden_layer_sizes=100, cv=10, n_jobs=-1):
    """
    :param X_train: Features
    :param y: Classes
    :param algorithm:
    :param max_iter:
    :param alpha: Learning rate
    :param hidden_layer_sizes:
    :param cv: kfold crossvalidation, k = 10
    :param n_jobs:
    :return:
    """

    clf = MLPClassifier(algorithm=algorithm, max_iter=max_iter, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                        random_state=1)
    print(cross_val_score(clf, X_train, y, cv=cv, n_jobs=n_jobs))
