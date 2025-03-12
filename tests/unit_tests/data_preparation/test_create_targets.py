# test_create_targets.py


import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from data_preparation.components.create_targets import CreateTargets


class TestCreateTargets(unittest.TestCase):

    @patch("data_preparation.components.create_targets.EnvLoader")
    def setUp(self, MockEnvLoader):
        """Set up an in-memory database and mock environment loader."""
        self.mock_env_loader = MockEnvLoader.return_value
        self.mock_env_loader.get.side_effect = lambda key: "mock_db_path" if key in [
            "base_source", "feature_source", "target_source", "BASE_DATABASE",
            "FEATURE_DATASE", "TARGET_DATABASE"] else None
        self.mock_env_loader.get.return_value = "/mock/config.py"

        self.mock_env_loader.load_config_module.return_value = {
            "main_target_module": "targets",
            "target_storage_map": "storage_map",
            "primary_key": ['id', 'obj'],
            "pair_query": {"BASE_DATABASE": "SELECT pair FROM pairs"}
        }

        self.mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'obj': ['a', 'b', 'c'],
            'value': ['data1', 'data2', 'data3']
        })

        self.create_targets = CreateTargets()

    @patch('data_preparation.components.create_targets.importlib.util.spec_from_file_location')
    @patch('data_preparation.components.create_targets.importlib.util.module_from_spec')
    def test_import_target_module(self, mock_module_from_spec, mock_spec_from_file_location):
        """Test that _import_target_module returns the target module."""
        mock_spec = MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        mock_spec.loader.exec_module.return_value = None
        mock_module.targets = MagicMock()

        targets_function = self.create_targets._import_target_module()
        self.assertEqual(targets_function, mock_module.targets)

    @patch('data_preparation.components.create_targets.importlib.util.spec_from_file_location') 
    @patch('data_preparation.components.create_targets.importlib.util.module_from_spec')
    def test_import_storage_map(self, mock_modules_from_spec, mock_spec_from_file_location):
        """Test that _import_storage_map imports the storage_map from the target script."""
        mock_spec = MagicMock()
        mock_spec_from_file_location.return_value = mock_spec
        mock_module = MagicMock()
        mock_modules_from_spec.return_value = mock_module
        mock_spec.loader.exec_module.return_value = None

        storage_map = self.create_targets._import_storage_map()
        self.assertEqual(storage_map, mock_module.storage_map)

    def test_store_original_columns(self):
        """Test that store_original_columns stores the original columns."""
        expected_columns = ['value']
        result_columns  = self.create_targets._store_original_columns(self.mock_df)
        self.assertEqual(expected_columns, result_columns )

    @patch("data_preparation.components.create_targets.pd.read_sql_query")
    @patch("data_preparation.components.create_targets.sqlite3.connect")
    def test_create_pairs_list(self, mock_sqlite_connect, mock_read_sql):
        """Test _create_pairs_list to ensure it correctly queries and returns pairs."""
        mock_conn = MagicMock()
        mock_sqlite_connect.return_value.__enter__.return_value = mock_conn

        mock_df = pd.DataFrame({"pair": ["EURUSD", "GBPUSD", "AUDUSD"]})
        mock_read_sql.return_value = mock_df

        pair_query = self.create_targets.config["pair_query"]
        result = self.create_targets._create_pairs_list(pair_query)

        expected_pairs = ["EURUSD", "GBPUSD", "AUDUSD"]
        self.assertEqual(result, expected_pairs) 

        mock_sqlite_connect.assert_called_once_with("mock_db_path")
        mock_read_sql.assert_called_once_with("SELECT pair FROM pairs", mock_conn)

    def test_create_batches_even_split(self):
        """Test _create_batches with an evenly divisible list."""
        pairs_list = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "EURGBP"]
        batch_size = 2

        result = list(self.create_targets._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"], ["AUDUSD", "USDJPY"], ["USDCAD", "EURGBP"]]

        self.assertEqual(result, expected)

    def test_create_batches_uneven_split(self):
        """Test _create_batches with an unevenly divisible list."""
        pairs_list = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD"]
        batch_size = 2

        result = list(self.create_targets._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"], ["AUDUSD", "USDJPY"], ["USDCAD"]]

        self.assertEqual(result, expected)

    def test_create_batches_single_batch(self):
        """Test _create_batches where batch_size is larger than list."""
        pairs_list = ["EURUSD", "GBPUSD"]
        batch_size = 5  # Larger than the list

        result = list(self.create_targets._create_batches(pairs_list, batch_size))
        expected = [["EURUSD", "GBPUSD"]]

        self.assertEqual(result, expected)

    def test_create_batches_empty_list(self):
        """Test _create_batches with an empty list."""
        pairs_list = []
        batch_size = 3

        result = list(self.create_targets._create_batches(pairs_list, batch_size))
        expected = []

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()