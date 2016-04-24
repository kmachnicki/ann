#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from src.consts import RANDOM_STATE


def select_features(X_train, y_train, k_best_features='all'):
    sel = SelectKBest(k=k_best_features).fit(X_train, y_train)
    return SelectKBest(k=k_best_features).fit_transform(X_train, y_train)


# sgd = stochastic gradient descent
def run_experiments_for_five_numbers_of_neurons(X_train, y, algorithm='sgd', max_iter=100, alpha=1e-6,
                                                learning_rate='constant',
                                                n_features='all', cv=10, n_jobs=-1,
                                                min_layer_size=None, max_layer_size=None):
    print("Running experiments.")
    if RANDOM_STATE:
        print("Use predefined random_state: True ({})".format(RANDOM_STATE))
    else:
        print("Use predefined random_state: False")
    print("Parameters: algorithm={}, max_iter={}, alpha={}, n_features={}, cv={}".format(algorithm, max_iter, alpha,
                                                                                         n_features, cv))
    X_selected = select_features(X_train, y, n_features)
    num_of_selected_features = len(X_selected[0])
    if num_of_selected_features == len(X_train[0]):
        print("Using all features.")
    else:
        print("Using {} selected features.".format(n_features))

    max_layer_size, min_layer_size = calc_layer_size(max_layer_size, min_layer_size, num_of_selected_features)

    best = (0, 0, 0)
    # for layer_size in range(min_layer_size, max_layer_size,
    #                         int((max_layer_size - min_layer_size) / 5)):
    layer_size=5
    print("layer_size =", layer_size)
    scores = run_experiment(X_selected, y, algorithm=algorithm, max_iter=max_iter, alpha=alpha,
                            learning_rate=learning_rate, hidden_layer_size=layer_size, cv=cv, n_jobs=n_jobs)

    print("{} vs {}".format(scores.mean() - scores.std() * 2, best[0] - best[1]))
    if scores.mean() - scores.std() * 2 > best[0] - best[1]:
        best = (scores.mean(), scores.std() * 2, layer_size)
        print("new best", best)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    print("Best (mean, std, layer_size): ", best)
    return best


def calc_layer_size(max_layer_size, min_layer_size, num_of_selected_features):
    if min_layer_size and not max_layer_size:
        max_layer_size = min_layer_size * 5
    elif not min_layer_size and not max_layer_size:
        min_layer_size = num_of_selected_features
        max_layer_size = min_layer_size * 5
    return max_layer_size, min_layer_size


def run_experiment(X_train, y, algorithm='sgd', max_iter=100, alpha=1e-6, hidden_layer_size=100,
                   learning_rate='constant', cv=10, n_jobs=-1):
    """
    :param learning_rate:
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
                        learning_rate=learning_rate, random_state=RANDOM_STATE)
    return cross_val_score(clf, X_train, y, cv=cv, n_jobs=n_jobs)
