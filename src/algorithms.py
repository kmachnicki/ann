#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold

import numpy as np
from collections import Counter
from time import time

from modules.elm import ELMClassifier
from src.consts import RANDOM_STATE
from src.helpers import AlgorithmOutput, ExperimentOutput


# sgd = stochastic gradient descent
def run_experiment(X, y, hidden_layer_size=100, n_features='all', algorithm='sgd', max_iter=100, alpha=1e-6,
                   learning_rate='constant', kfold=10, activation_func='multiquadric'):
    X = np.array(X)
    y = np.array(y)

    counter = Counter()

    bp_scores, bp_fit_times, bp_score_times, elm_scores, elm_fit_times, elm_score_times = [], [], [], [], [], []

    for train_set, test_set in KFold(n_folds=kfold).split(X, y):
        X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
        indices = get_selected_features_indices(X_train, y_train, k_best_features=n_features)
        X_train = [row[indices] for row in X_train]
        X_test = [row[indices] for row in X_test]
        counter.update(indices)

        bp_result = run_bp(X_train, y_train, X_test, y_test,
                            algorithm=algorithm, max_iter=max_iter, alpha=alpha,
                            learning_rate=learning_rate, hidden_layer_size=hidden_layer_size)

        bp_scores.append(bp_result.score)
        bp_fit_times.append(bp_result.fit_time)
        bp_score_times.append(bp_result.score_time)

        elm_result = run_elm(X_train, y_train, X_test, y_test,
                              n_hidden=hidden_layer_size, activation_func=activation_func)

        elm_scores.append(elm_result.score)
        elm_fit_times.append(elm_result.fit_time)
        elm_score_times.append(elm_result.score_time)

    return ExperimentOutput(AlgorithmOutput(np.mean(bp_scores), np.mean(bp_fit_times), np.mean(bp_score_times)),
                            AlgorithmOutput(np.mean(elm_scores), np.mean(elm_fit_times), np.mean(elm_score_times)),
                            counter)


def get_selected_features_indices(X, y, k_best_features='all'):
    return SelectKBest(k=k_best_features).fit(X, y).get_support(indices=True)


def run_bp(X_train, y_train, X_test, y_test, algorithm='sgd', max_iter=200, alpha=1e-6, hidden_layer_size=100,
                    learning_rate='constant'):

    '''
    Back propagation algorithms for the single layer feedforward network (SLFN)
    '''

    clf = MLPClassifier(algorithm=algorithm, max_iter=max_iter, alpha=alpha, hidden_layer_sizes=hidden_layer_size,
                        learning_rate=learning_rate, random_state=RANDOM_STATE)

    fit_start_time = time()
    clf.fit(X_train, y_train)
    fit_time = time() - fit_start_time

    score_start_time = time()
    score = clf.score(X_test, y_test)
    score_time = time() - score_start_time

    return AlgorithmOutput(score, fit_time, score_time)


def run_elm(X_train, y_train, X_test, y_test, n_hidden=100, activation_func='multiquadric'):

    '''
    Extreme learning machine algorithm for the single layer feedforward network (SLFN)
    '''

    elmc = ELMClassifier(n_hidden=n_hidden, activation_func=activation_func)

    fit_start_time = time()
    elmc.fit(X_train, y_train)
    fit_time = time() - fit_start_time

    score_start_time = time()
    score = elmc.score(X_test, y_test)
    score_time = time() - score_start_time

    return AlgorithmOutput(score, fit_time, score_time)
