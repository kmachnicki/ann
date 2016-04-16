#!/usr/bin/python

from sklearn.neural_network import MLPClassifier
import random


def check_results(training_set, output, correct_answer):
    fail_index = [i for i, j in enumerate(output) if j != correct_answer]
    if fail_index:
        print("Nie uzyskano klasy", correct_answer, "w nastepujacych przypadkach:")
        for index in fail_index:
            print(training_set[index])
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

clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(1000), random_state=1)
classifier = clf.fit(X, y)

test_1 = [[i, i] for i in range(1000)]
test_2 = [[i, i+random.randint(1, 200)] for i in range(1000)]
test_3 = [[i, i-random.randint(1, 200)] for i in range(1000)]

check_results(test_1, classifier.predict(test_1), 1)
check_results(test_2, classifier.predict(test_2), 2)
check_results(test_3, classifier.predict(test_3), 3)

print("Done.")



