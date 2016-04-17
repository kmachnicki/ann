#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest


def select_features(X_train, y_train, k_best_features=10):
    return SelectKBest(k=k_best_features).fit_transform(X_train, y_train)


def run_experiments_for_five_numbers_of_neurons(X_train, y, algorithm='l-bfgs', max_iter=100, alpha=1e-6,
                                                hidden_layer_size=100, cv=10, n_jobs=-1):
    for layer_size in range(50, 550, 100):
        print("Running experiment for layer_size =", layer_size)
        run_experiment(X_train, y, algorithm=algorithm, max_iter=max_iter, alpha=alpha,
                       hidden_layer_size=hidden_layer_size, cv=cv, n_jobs=n_jobs)


def run_experiment(X_train, y, algorithm='l-bfgs', max_iter=100, alpha=1e-6, hidden_layer_size=100, cv=10, n_jobs=-1):
    """
    :param X_train: Features
    :param y: Classes
    :param algorithm:
    :param max_iter:
    :param alpha: parameter for regularization term, aka penalty term,
            that combats overfitting by constraining the size of the weights.
            Increasing alpha may fix high variance (a sign of overfitting) by encouraging smaller weights,
            resulting in a decision boundary plot that appears with lesser curvatures.
            Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger weights,
            potentially resulting in a more complicated decision boundary.
    :param hidden_layer_size:
    :param cv: kfold crossvalidation, k = 10
    :param n_jobs:
    :return:
    """

    clf = MLPClassifier(algorithm=algorithm, max_iter=max_iter, alpha=alpha, hidden_layer_sizes=hidden_layer_size,
                        random_state=1)
    print(cross_val_score(clf, X_train, y, cv=cv, n_jobs=n_jobs))
