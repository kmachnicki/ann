#!/usr/bin/python
import csv


class DataSet:
    def __init__(self, X=None, y=None):
        self._X = X if X else []
        self._y = y if y else []

    def extract_from_csv(self, csv_file):
        self._X = []
        self._y = []
        reader = csv.reader(csv_file)
        self._skip_header(csv_file, reader)
        for row in reader:
            extracted_features = [float(i) for i in row[:-1]]
            self._X.append(extracted_features)
            extracted_class = int(row[-1])
            self._y.append(extracted_class)
        return self._X, self._y

    @staticmethod
    def _skip_header(csv_file, reader):
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)
        if has_header:
            next(reader)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
