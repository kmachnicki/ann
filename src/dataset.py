#!/usr/bin/python

from csv import reader


class DataSet:
    def __init__(self, X=None, y=None, cases=None):
        self._X = X
        self._y = y
        self._cases = cases
        self._check_features_sizes()

    def extract_from_csv(self, csv_file):
        self._X = []
        self._y = []
        self._cases = []
        csv_reader = reader(csv_file, delimiter=';')
        self._skip_header(csv_reader)
        for row in csv_reader:
            extracted_features = [float(i.replace(',', '.')) for i in row[1:-1]]
            self._X.append(extracted_features)
            extracted_class = str(row[-1])
            self._y.append(extracted_class)
            extracted_case = str(row[0])
            self._cases.append(extracted_case)

        self._check_features_sizes()
        return self._X, self._y, self._cases

    @staticmethod
    def _skip_header(csv_reader):
        next(csv_reader)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def cases(self):
        return self._cases

    @property
    def number_of_features(self):
        return len(self.X[0])

    def _check_features_sizes(self):
        if self.X and self.y:
            prev_len = len(self.X[0])
            for features in self.X[1:]:
                if len(features) != prev_len:
                    raise RuntimeError("Rows sizes mismatch. Check your csv file.")
                prev_len = len(features)
        elif any([self.X, self.y]):
            raise RuntimeError("Wrong initial data")
