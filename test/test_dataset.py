from os import path

from unittest import TestCase
from src.dataset import DataSet

DATA_SETS_DIR = path.join(path.dirname(__file__), "data_sets")


class TestDataSet(TestCase):
    def setUp(self):
        self.init_features = [[1.5, -2.5, 3.], [-3., 0.5, 0.]]
        self.init_classes = ["G3", "G1"]
        self.init_cases = ["case1", "case4"]

        self.expected_extracted_features = [[1.7, 3., 0.09], [-5., -1.12, 0.]]
        self.expected_extracted_classes = ["G2", "G3"]
        self.expected_extracted_cases = ["1666-3b", "204"]

        self.data_set_dir = path.join(path.dirname(__file__), "data_sets")

        self.data_set = DataSet(X=self.init_features, y=self.init_classes, cases=self.init_cases)

    def check_extracted(self):
        self.assertListEqual(self.expected_extracted_features, self.data_set.X)
        self.assertListEqual(self.expected_extracted_classes, self.data_set.y)
        self.assertListEqual(self.expected_extracted_cases, self.data_set.cases)

    def test_should_initialize_properly(self):
        self.assertListEqual(self.init_features, self.data_set.X)
        self.assertListEqual(self.init_classes, self.data_set.y)
        self.assertListEqual(self.init_cases, self.data_set.cases)

    def test_should_extract_features_and_classes_from_csv_with_header(self):
        with open(path.join(DATA_SETS_DIR, "mock_data_set_with_header.csv"),
                  "r", newline='', encoding="utf8") as csv_file:
            self.data_set.extract_from_csv(csv_file)
        self.check_extracted()

    def test_should_raise_error_on_feature_size_mismatch(self):
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]])
        with self.assertRaises(RuntimeError):
            DataSet(y=[1])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]], y=[1])
        with open(path.join(DATA_SETS_DIR, "mock_data_set_corrupted.csv"),
                  "r", newline='', encoding="utf8") as csv_file:
            with self.assertRaises(RuntimeError):
                self.data_set.extract_from_csv(csv_file)

    def test_should_return_number_of_features(self):
        self.assertEqual(len(self.init_features[0]), self.data_set.number_of_features)
