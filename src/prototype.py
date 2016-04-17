#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
import random
from src.dataset import DataSet


def check_results(training_set, output, correct_answer):
    fail_index = [i for i, j in enumerate(output) if j != correct_answer]
    if fail_index:
        print("Nie uzyskano klasy", correct_answer, "w nastepujacych przypadkach:")
        for index in fail_index:
            print(training_set[index])


def create_data():
    X = []
    y = []

    # "f(x) = x"
    for i in range(300):
        X.append([i, i])
        y.append(1)

    # "f(x) > x"
    for i in range(300):
        X.append([i, i+random.randint(1, 100)])
        y.append(2)

    # "f(x) < x"
    for i in range(300):
        X.append([i, i-random.randint(1, 100)])
        y.append(3)

    return X, y


def create_test_data():
    test_1 = [[i, i] for i in range(1000)]
    test_2 = [[i, i + random.randint(1, 200)] for i in range(1000)]
    test_3 = [[i, i - random.randint(1, 200)] for i in range(1000)]
    return test_1, test_2, test_3


def main():
    ds = DataSet()
    with open("../stopien_zlosliwosci.csv", newline='') as csv_file:
        ds.extract_from_csv(csv_file)

    clf = MLPClassifier(algorithm='l-bfgs', max_iter=1000, alpha=1e-6, hidden_layer_sizes=(1500), random_state=1)
    classifier = clf.fit(ds.X, ds.y)

    test_1, test_2, test_3 = create_test_data()

    check_results(test_1, classifier.predict(test_1), 1)
    check_results(test_2, classifier.predict(test_2), 2)
    check_results(test_3, classifier.predict(test_3), 3)

    print("Done.")

if __name__ == '__main__':
    main()

