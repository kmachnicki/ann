#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from src.consts import RANDOM_STATE
from sklearn.model_selection import KFold
import numpy as np
from collections import Counter


def get_selected_features_indices(X, y, k_best_features='all'):
    return SelectKBest(k=k_best_features).fit(X, y).get_support(indices=True)


# sgd = stochastic gradient descent
def run_experiment(X, y, algorithm='sgd', max_iter=100, alpha=1e-6,
                   learning_rate='constant', n_features='all', hidden_layer_size=100, kfold=10):
    X = np.array(X)
    y = np.array(y)
    mean = 0
    counter = Counter()
    for train_set, test_set in KFold(n_folds=kfold).split(X, y):
        X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
        indices = get_selected_features_indices(X_train, y_train, k_best_features=n_features)
        counter.update(indices)
        X_train = [row[indices] for row in X_train]
        X_test = [row[indices] for row in X_test]
        mean += calculate_score(X_train, y_train, X_test, y_test, algorithm=algorithm, max_iter=max_iter, alpha=alpha,
                                learning_rate=learning_rate, hidden_layer_size=hidden_layer_size)
    return mean / kfold, counter


def calculate_score(X_train, y_train, x_test, y_test, algorithm='sgd', max_iter=200, alpha=1e-6, hidden_layer_size=100,
                    learning_rate='constant'):
    clf = MLPClassifier(algorithm=algorithm, max_iter=max_iter, alpha=alpha, hidden_layer_sizes=hidden_layer_size,
                        learning_rate=learning_rate, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf.score(x_test, y_test)
