#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import numpy as np
from collections import Counter
from time import time

from modules.elm import ELMClassifier
from src.consts import RANDOM_STATE, N_RUNS, K_BEST_FEATURES, N_FOLDS, BP_ALGORITHM, BP_MAX_ITER, BP_ALPHA,\
    BP_LEARNING_RATE, ELM_ACTIVATION_FUNC
from src.helpers import Sample, ExperimentOutput, ExperimentWrapper


# sgd = stochastic gradient descent
def run_experiment(X, y, hidden_layer_size, n_features, n_runs=N_RUNS, algorithm=BP_ALGORITHM,
                   max_iter=BP_MAX_ITER, alpha=BP_ALPHA, learning_rate=BP_LEARNING_RATE,
                   n_folds=N_FOLDS, activation_func=ELM_ACTIVATION_FUNC):
    X = np.array(X)
    y = np.array(y)

    counter = Counter()
    experiment_bp = ExperimentWrapper()
    experiment_elm = ExperimentWrapper()

    for i in range(n_runs):
        kfolds_bp = ExperimentWrapper()
        kfolds_elm = ExperimentWrapper()

        for train_set, test_set in KFold(n_folds=n_folds).split(X, y):
            X_train, X_test, y_train, y_test = X[train_set], X[test_set], y[train_set], y[test_set]
            indices = get_selected_features_indices(X_train, y_train, k_best_features=n_features)
            X_train = [row[indices] for row in X_train]
            X_test = [row[indices] for row in X_test]
            counter.update(indices)

            kfolds_bp.add_sample(
                run_bp(X_train, y_train, X_test, y_test,
                       algorithm=algorithm, max_iter=max_iter, alpha=alpha,
                       learning_rate=learning_rate, hidden_layer_size=hidden_layer_size))

            kfolds_elm.add_sample(
                run_elm(X_train, y_train, X_test, y_test,
                        n_hidden=hidden_layer_size, activation_func=activation_func))

        kfolds_bp_samples = kfolds_bp.samples()
        experiment_bp.add_sample(
            Sample(np.mean(kfolds_bp_samples.scores),
                   np.mean(kfolds_bp_samples.fit_times),
                   np.mean(kfolds_bp_samples.score_times),
                   kfolds_bp_samples.conf_matrices[0]))

        kfolds_elm_samples = kfolds_elm.samples()
        experiment_elm.add_sample(
            Sample(np.mean(kfolds_elm_samples.scores),
                   np.mean(kfolds_elm_samples.fit_times),
                   np.mean(kfolds_elm_samples.score_times),
                   kfolds_elm_samples.conf_matrices[0]))

    return ExperimentOutput(experiment_bp.samples(),
                            experiment_elm.samples(),
                            counter)


def get_selected_features_indices(X, y, k_best_features=K_BEST_FEATURES):
    return SelectKBest(k=k_best_features).fit(X, y).get_support(indices=True)


def run_bp(X_train, y_train, X_test, y_test, algorithm, max_iter, alpha, hidden_layer_size, learning_rate):

    '''
    Back propagation algorithms for the single layer feedforward network (SLFN)
    '''

    clf = MLPClassifier(algorithm=algorithm, max_iter=max_iter, alpha=alpha, hidden_layer_sizes=hidden_layer_size,
                        learning_rate=learning_rate, random_state=RANDOM_STATE, activation='logistic')

    fit_start_time = time()
    clf.fit(X_train, y_train)
    fit_time = time() - fit_start_time
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    score_start_time = time()
    score = clf.score(X_test, y_test)
    score_time = time() - score_start_time
    print(conf_matrix)

    return Sample(score, fit_time, score_time, conf_matrix)


def run_elm(X_train, y_train, X_test, y_test, n_hidden, activation_func):

    '''
    Extreme learning machine algorithm for the single layer feedforward network (SLFN)
    '''

    elmc = ELMClassifier(n_hidden=n_hidden, activation_func=activation_func)


    fit_start_time = time()
    elmc.fit(X_train, y_train)
    fit_time = time() - fit_start_time

    conf_matrix = confusion_matrix(elmc.predict(X_test), y_test)

    score_start_time = time()
    score = elmc.score(X_test, y_test)
    score_time = time() - score_start_time

    return Sample(score, fit_time, score_time, conf_matrix)
