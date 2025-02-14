# eval_inference.py


import importlib.util
import numpy as np
import json
import re
import glob

import os
# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from utils import EnvLoader, public_method


class Eval:
    def __init__(self):        
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("EVALUATE_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        self.backtest_config = self.config.get("backtest")
        
    #-----------------------------------------------------#
    # General Utils
    #-----------------------------------------------------#
        
    def _import_function(self, module_path, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified function dynamically
        function = getattr(module, function_name)
        
        return function
    
    def _initial_bias(self, model_args, weight_dict_path):
        """
        Compute initial bias for model output layer based on class distribution.
        
        Args:
            mode: self.backtest_config or self.forwardtest_config
        
        Returns:
            np.array: Initial bias values for each target class.
        """
        # Check if initial_bias is enabled in model_args
        if not model_args.get("initial_bias", False):
            return None  # No initial bias required
    
        # Load class weights from the JSON file
        if not os.path.exists(weight_dict_path):
            raise FileNotFoundError(f"Weight dictionary not found at: {self.weight_dict_path}")
    
        with open(weight_dict_path, "r") as f:
            class_weights = json.load(f)
    
        # Convert keys to integers (as JSON keys are stored as strings)
        class_weights = {int(k): v for k, v in class_weights.items()}
    
        # Compute class probabilities
        total_weight = sum(class_weights.values())
        class_probs = {k: v / total_weight for k, v in class_weights.items()}
    
        # Compute log-odds for initial bias
        epsilon = 1e-8  # Small value to prevent log(0)
        initial_bias = np.log([class_probs[i] / (1 - class_probs[i] + epsilon) for i in sorted(class_probs)])
    
        print("Computed Initial Bias.")
        return initial_bias   
    
    #-----------------------------------------------------#
    # Evaluate on preprocessed backtest data
    #-----------------------------------------------------#   
    
    # backtest eval utils --------------------------------#
    
    def _model_config(self):
        """
        Return the original iteration's config.py dict.
        """
        config_path = os.path.join(
            self.backtest_config["dir"], self.backtest_config["model_config_src"])
        model_config = self._import_function(
            config_path, self.backtest_config["config_dict_name"])
        return model_config
    
    def _group_pred_files(self, data_dir, model_config, feature_category, mode):
        """
        Go through the prepped data directory and group feature and target
        sets by their number.
        """
        tfrecord_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tfrecord')])
        
        if mode in ['train', 'val', 'test']:
            feature_files = [f for f in tfrecord_files if f.startswith(feature_category) and mode in f]
            target_files = [f for f in tfrecord_files if "targets" in f and mode in f]
        elif mode == 'full':
            feature_files = [f for f in tfrecord_files if "targets" not in f]
            target_files = [f for f in tfrecord_files if "targets" in f]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'full', 'train', 'val', or 'test'.")

        file_dict = {}

        for f in feature_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num not in file_dict:
                    file_dict[num] = {'features': [], 'targets': []}
                file_dict[num]['features'].append(os.path.join(data_dir, f))

        for f in target_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num in file_dict:  # Ensuring targets align with features
                    file_dict[num]['targets'].append(os.path.join(data_dir, f))

        return file_dict 
    
    def _load_feature_metadata(self, feature_category, data_dir):
        """
        Load feature metadata from the corresponding JSON file dynamically.
        - Searches for files matching the expected pattern.
        - If exact match is missing, searches for similar files.
        """
        # Define expected JSON filename pattern
        expected_filename = f"{feature_category}_feature_description.json"
        
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
        # Check for exact match
        json_file_path = os.path.join(data_dir, expected_filename)
        if json_file_path in json_files:
            pass  # File exists, proceed with loading
    
        # If exact match is missing, try to find a closely matching file
        else:
            matching_files = [f for f in json_files if feature_category in os.path.basename(f)]
            
            if len(matching_files) == 1:
                json_file_path = matching_files[0]  # Use closest match
            elif len(matching_files) > 1:
                raise FileNotFoundError(f"Multiple feature description JSON files found for {feature_category}: {matching_files}. "
                                        f"Please specify the correct one.")
            else:
                raise FileNotFoundError(f"Feature description JSON not found for {feature_category}. "
                                        f"Expected: {expected_filename}, but found: {json_files}")
    
        # Load JSON metadata
        with open(json_file_path, "r") as f:
            metadata = json.load(f)
    
        return metadata
    
    def _load_target_metadata(self, data_dir):
        """
        Load target metadata from the corresponding JSON file.
        """
        json_file = os.path.join(data_dir, "targets_feature_description.json")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Target description JSON not found: {json_file}")

        with open(json_file, "r") as f:
            metadata = json.load(f)
            
            # Auto-fix incorrect data types
        for key, value in metadata.items():
            if value == "int64":
                metadata[key] = tf.io.FixedLenFeature([], tf.int64)  # Ensure correct int64 format
            elif value == "float32":
                metadata[key] = tf.io.FixedLenFeature([], tf.float32)  # Ensure correct float32 format
            elif value == "string":
                metadata[key] = tf.io.VarLenFeature(tf.string)  # Use VarLenFeature for string values
            elif value == "object":
                metadata[key] = tf.io.VarLenFeature(tf.string) # Comvert objects to strings
            else:
                raise ValueError(f"Unsupported data type '{value}' for key '{key}' in target metadata.")

        return metadata
    
    def _parse_tfrecord_fn(self, example_proto, feature_metadata, category,
                           feature_categories):
        """
        Parses a TFRecord example into features while ensuring proper reshaping.
        """
        feature_description = {
            key: tf.io.FixedLenFeature([], tf.float32)
            for key in feature_metadata.keys()
        }
        # Parse only the feature part (exclude the target)
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
        # Convert feature dictionary to tensor
        feature_tensor = tf.stack(list(parsed_example.values()), axis=0)
    
        # Apply reshaping if required
        category_config = feature_categories[category]
        if category_config["reshape"]:
            feature_tensor = tf.reshape(feature_tensor, category_config["shape"])
    
        return feature_tensor
    
    @tf.autograph.experimental.do_not_convert
    def _load_tfrecord_dataset(self, feature_files, target_files, feature_categories,
                               data_dir, batch_size):
        """
        Load and process datasets from multiple aligned feature and target TFRecord files.
        Ensures correct structure for model input.
        """
        feature_metadata = {category: self._load_feature_metadata(category, data_dir) for category in feature_categories}
        target_metadata = self._load_target_metadata(data_dir)
        
        feature_datasets = {
            category: tf.data.TFRecordDataset(files).map(
                lambda x: self._parse_tfrecord_fn(x, feature_metadata[category], 
                                                  category, feature_categories),
                num_parallel_calls=tf.data.AUTOTUNE
            ) for category, files in feature_files.items()
        }
    
        # Load and parse target dataset separately (return full data + `target`)
        def parse_target_fn(x):
            parsed = tf.io.parse_single_example(x, target_metadata)
            return parsed, parsed['target']  # Return full parsed targets & just `target` for training
    
        target_dataset = tf.data.TFRecordDataset(target_files).map(
            parse_target_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
    
        # Extract only the target column for training
        target_labels_dataset = target_dataset.map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE)
    
        # Zip full target data separately for saving CSV
        full_target_dataset = target_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    
        # Convert feature dictionary to tuple (preserves ordering)
        feature_tuple_dataset = tf.data.Dataset.zip((
            tuple(feature_datasets.values()),  # Tuple of feature inputs
            target_labels_dataset  # Only target label for training
        ))
    
        # Batch and prefetch for efficiency
        feature_tuple_dataset = feature_tuple_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        full_target_dataset = full_target_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        return feature_tuple_dataset, full_target_dataset, list(target_metadata.keys())
    
    # backtest eval main functions -----------------------#
    
    def _load_model(self):
        """
        Use the backtest config to source the model weights, architecture,
        and arguements.
        """
        # Create the args to import model architecture
        model_modules_path = os.path.join(
            self.backtest_config["dir"], self.backtest_config["create_model_module_src"])
        model_function = self.backtest_config["model_module"]
        
        # Load the create_model function for the iteration
        create_model = self._import_function(
            model_modules_path, model_function)
        
        # Load the model args
        model_config = self._model_config()
        model_args = model_config[self.backtest_config["model_args"]]
        weight_dict_path = model_config["weight_dict_path"]
        
        # Compute initial bias if needed
        initial_bias = self._initial_bias(model_args, weight_dict_path)
        
        # Pass model arguments, updating the initial_bias field if applicable
        if initial_bias is not None:
            model_args["initial_bias"] = initial_bias
        
        # Pass the model's parameters from the config dict
        if callable(create_model):
            model = create_model(**model_args)
        else:
            raise TypeError("create_model is not callable")
            
        return model
    
    def _load_weights(self, model):
        """
        Use the backtest config's specified weights file to load into model
        architecture.
        """
        weights_path = os.path.join(
            self.backtest_config["dir"], self.backtest_config["model_weights"])
        
        model.load_weights(weights_path)
        print("Weights loaded successfully.")
        return model
    
    def _backtest_predict(self, model, mode):
        """
        Group the prepped data, load them as datasets, predict, and save 
        outcomes. 
        
        Args:
            model (obj): tf model with loaded weights
            mode (str): 'full', 'train', 'val', or 'test' specify which 
                        datasets to make predictions with
        """
        # Loop through all numbered datasets (0, 1, 2, etc.)
        model_config = self._model_config()
        data_dir = model_config["data_dir"]
        batch_size = model_config["batch_size"]
        feature_categories = model_config["feature_categories"]
        
        # Group the prepped data so feature sets are bundled with corresponding
        # target sets
        predict_groups = {category: self._group_pred_files(data_dir, model_config, category, mode)
                          for category in feature_categories}
        
        for num in sorted(predict_groups[next(iter(feature_categories))].keys()):
            feature_files = {
                category: [
                    f for f in predict_groups[category].get(num, {}).get('features', []) 
                    if category in os.path.basename(f)  # Ensures files are correctly matched
                ]
                for category in feature_categories
            }
            target_files = predict_groups[
                next(iter(feature_categories))].get(num, {}).get('targets', [])

            # Ensure all feature categories are present
            if any(not files for files in feature_files.values()) or not target_files:
                continue  # Skip if any feature category is missing

            print(f"Predicting set {num}")
            
            # Load dataset with all feature sets
            dataset, full_target_dataset, target_column_names = (
                self._load_tfrecord_dataset(feature_files, target_files, feature_categories,
                                            data_dir, batch_size))
            
            predictions = model.predict(dataset)
        
    @public_method
    def backtest_results(self, mode='full'):
        """
        Load data, model, and respective specs from the iteration specified in
        config. Perform model.predict, save results, and report metrics.
        
        Args:
            mode (str): 'full', 'train', 'val', or 'test' specify which 
                        datasets to make predictions with
        """
        model = self._load_model()
        model = self._load_weights(model)
        self._backtest_predict(model, mode)
        
    #-----------------------------------------------------#
    # Evaluate on preprocessed forward test data
    #-----------------------------------------------------#
        