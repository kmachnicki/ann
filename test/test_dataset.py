from unittest import TestCase
from src.dataset import DataSet


class TestDataSet(TestCase):
    def setUp(self):
        self.init_features = [[1.5, 2.5, 3.5], [3.5, 2.5, 1.5]]
        self.init_classes = [1, 0]
        self.dataSet = DataSet(X=self.init_features, y=self.init_classes)

        self.expected_extracted_features = [[1., 2., 3.], [3., 2., 1.]]
        self.expected_extracted_classes = [1, 0]

    def check_extracted(self):
        self.assertListEqual(self.expected_extracted_features, self.dataSet.X)
        self.assertListEqual(self.expected_extracted_classes, self.dataSet.y)

    def test_should_initialize_properly(self):
        self.assertListEqual(self.init_features, self.dataSet.X)
        self.assertListEqual(self.init_classes, self.dataSet.y)

    def test_should_extract_features_and_classes_from_csv_with_header(self):
        with open("mock_data_set_with_header.csv") as csv_file:
            self.dataSet.extract_from_csv(csv_file)
        self.check_extracted()

    def test_should_extract_features_and_classes_from_csv_without_header(self):
        with open("mock_data_set_without_header.csv") as csv_file:
            self.dataSet.extract_from_csv(csv_file)
        self.check_extracted()

    def test_should_raise_error_on_feature_size_mismatch(self):
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]])
        with self.assertRaises(RuntimeError):
            DataSet(y=[1])
        with self.assertRaises(RuntimeError):
            DataSet(X=[[1], [1, 2]], y=[1])
        with open("mock_data_set_corrupted.csv") as csv_file:
            with self.assertRaises(RuntimeError):
                self.dataSet.extract_from_csv(csv_file)

    def test_should_return_number_of_features(self):
        self.assertEqual(len(self.init_features[0]), self.dataSet.number_of_features)
