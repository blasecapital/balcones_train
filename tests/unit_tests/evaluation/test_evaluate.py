# test_evaluate.py


import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os

import numpy as np

from evaluation.components.evaluate import Eval


class TestEval(unittest.TestCase):

    def setUp(self):
        self.eval = Eval()

    def test_import_function(self):
        """Test that `_import_function` correctly loads a function from a dynamically imported module."""
        function_code = """
def test_function():
    return "Hello, World!"
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "test_module.py")
            
            # Write test function to a temporary module file
            with open(module_path, "w") as f:
                f.write(function_code)

            # Set module_path in PrepData
            self.eval.module_path = module_path  

            # Call `_import_function` and check if it correctly loads `test_function`
            imported_function = self.eval._import_function(module_path, "test_function")
            self.assertEqual(imported_function(), "Hello, World!")

    @patch("builtins.open", new_callable=mock_open, read_data='{"0": 0.5, "2": 3.5, "1": 3.5}')
    @patch("os.path.exists", return_value=True)
    def test_initial_bias(self, mock_exists, mock_file):
        """Test initial bias is correctly calculated."""
        model_args = {
            "initial_bias": True 
        }
        expected_result = np.array([-2.639057, -0.133531, -0.133531])
        result = self.eval._initial_bias(model_args, weight_dict_path="test_path")
        np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()