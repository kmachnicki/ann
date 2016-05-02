from os import path

from unittest import TestCase
from src.dataset import DataSet

DATA_SETS_DIR = path.join(path.dirname(__file__), "data_sets")


class TestDataSet(TestCase):
    def setUp(self):
        self.init_column_names = ["feat_001", "X-coord", "h,std,dev", "Grade"]
        self.init_features = [[0., 0.1, 3.], [0., 0.2, 0.], [0., 0.3, 0.5]]
        self.init_classes = ["G3", "G1", "G1"]

        self.expected_extracted_column_names = self.init_column_names
        self.expected_extracted_features = [[1.7, 3., 0.09], [-5., -1.12, 0.]]
        self.expected_extracted_classes = ["G2", "G3"]

        self.data_set_dir = path.join(path.dirname(__file__), "data_sets")

        self.data_set = DataSet(X=self.init_features, y=self.init_classes, col_names=self.init_column_names)

    def check_extracted(self):
        self.assertListEqual(self.expected_extracted_column_names, self.data_set.col_names)
        self.assertListEqual(self.expected_extracted_features, self.data_set.X)
        self.assertListEqual(self.expected_extracted_classes, self.data_set.y)

    def test_should_initialize_properly(self):
        self.assertListEqual(self.init_column_names, self.data_set.col_names)
        self.assertListEqual(self.init_features, self.data_set.X)
        self.assertListEqual(self.init_classes, self.data_set.y)

    def test_should_extract_features_and_classes_from_csv_with_header(self):
        with open(path.join(DATA_SETS_DIR, "mock_data_set_with_header.csv"),
                  "r", newline='', encoding="utf8") as csv_file:
            self.data_set.extract_from_csv(csv_file)
        self.check_extracted()

    def test_should_raise_error_on_feature_size_mismatch(self):
        with self.assertRaises(RuntimeError) as cm:
            DataSet(X=[[1], [1, 2]])
        with self.assertRaises(RuntimeError):
            DataSet(y=[1])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]], y=[1])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]], y=[1, 1])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]], y=[1, 1, 1])
        with self.assertRaises(RuntimeError):
            DataSet(col_names=["a"])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1, 2], [1, 2]], y=[1, 1], col_names=["a"])
        with open(path.join(DATA_SETS_DIR, "mock_data_set_corrupted.csv"),
                  "r", newline='', encoding="utf8") as csv_file:
            with self.assertRaises(RuntimeError):
                self.data_set.extract_from_csv(csv_file)

    def test_should_return_number_of_features(self):
        self.assertEqual(len(self.init_features[0]), self.data_set.number_of_features)

    def test_should_return_column_name(self):
        for index, element in enumerate(self.init_column_names):
            self.assertEqual(element, self.data_set.col_names[index])

    def test_should_create_ranking(self):
        ranking = [2, 1, 0]
        self.assertListEqual(self.data_set.create_features_ranking(use_names=False), ranking)
