# eval_inference.py


import importlib.util
import numpy as np
import pandas as pd
import sqlite3

import os
# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from config import config
from utils import EnvLoader, public_method


class EvalInference:
    def __init__(self, training_data: dict, model_name):
        # The initialized object
        self.training_data = training_data
        self.model_name = model_name
        
        self.env_loader = EnvLoader()
        
        self.eval_dict = config
        
        # Config specs
        self.original_config = self._load_original_config()
        
        # Initialize config attributes
        self.model_specs = self.original_config["model_specs"][self.model_name]
        
    def _import_function(self, module_path, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified filter function dynamically
        filter_function = getattr(module, function_name)
        
        return filter_function
    
    def _load_original_config(self):
        """
        Dynamically load the original `config` module from the requirements directory.

        Returns:
            dict: The `config` dictionary from the original file.
        
        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ImportError: If the module cannot be loaded.
        """
        original_config_path = self.eval_dict[self.model_name]["requirements_directory"]
        
        if not os.path.exists(original_config_path):
            raise FileNotFoundError(f"The configuration file does not exist at: {original_config_path}")
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("config", original_config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Return the config dictionary
        return config_module.config
        
    def _load_model_and_weights(self):
        """
        Load the model architecture and its weights.
    
        Returns:
            model: The model with loaded weights.
        """
        # Load model architecture
        input_dim = self.training_data['features'][self.model_name]['train'].shape[1]
        spec = self.model_specs
        model_function = self._import_function(
            spec["model_modules_path"], spec["model_function"]
        )
        model = model_function(input_dim)
    
        # Load weights
        weights_path = self.eval_dict[self.model_name]["model_path"]
        model.load_weights(weights_path)
    
        return model
    
    def _decode_predictions(self):
        """
        """
        return print("This will be a great feature!")

    def _prepare_for_export(self):
        """
        """
        
    def _export_eval_inferences(self, export_df):
        """
        """
        table_name = self.eval_dict[self.model_name]["export_predictions"][0]
        source_type = self.eval_dict[self.model_name]["export_predictions"][1]
        db_path = self.env_loader.get(source_type)
        if_exists='replace'
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            print(f"Connected to database at {db_path}")
            
            # Export DataFrame to SQLite
            export_df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)
            print(f"DataFrame successfully exported to table '{table_name}' in the database.")
        
        except Exception as e:
            print(f"An error occurred while exporting DataFrame: {e}")
        
        finally:
            # Ensure the connection is closed
            conn.close()
            print("Database connection closed.")
        
    @public_method
    def eval_inference(self):
        """
        Make and store predictions.
        """
        test_features = self.training_data['features'][self.model_name]['test']
        test_targets = self.training_data['targets'][self.model_name]['test']
        model = self._load_model_and_weights()
        predictions = model.predict(test_features)
        
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        prediction_confidence = np.max(predictions, axis=1)
        
        export_df = pd.DataFrame({
            'predicted_target': predicted_classes,
            'confidence': prediction_confidence,
            'ground_truth': test_targets
            })
        
        self._export_eval_inferences(export_df)
        
        print("Successfully completed eval inference:", export_df)
        