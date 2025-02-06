# train.py


import importlib.util
import gc
import json
import re
import glob
import os
from datetime import datetime
import csv
import shutil

# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np

from utils import EnvLoader, public_method


class Train:
    def __init__(self):
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Initialize config attributes
        self.model_modules_path = self.config.get("model_modules_path")
        self.model_args = self.config.get("model_args")
        self.initial_bias_path = self.config.get("initial_bias_path")
        self.model_function = self.config.get("model_function")
        self.custom_loss = self.config.get("custom_loss")
        self.loss = self.config.get("loss")
        self.metrics = self.config.get("metrics")
        self.learning_rate_schedule = self.config.get("learning_rate_schedule")
        self.optimizer = self.config.get("optimizer")
        self.data_dir = self.config.get("data_dir")
        self.epochs = self.config.get("epochs")
        self.batch_size = self.config.get("batch_size")
        self.early_stopping = self.config.get("early_stopping")
        self.checkpoint = self.config.get("checkpoint")
        self.feature_categories = self.config.get("feature_categories")
        self.weight_dict_path = self.config.get("weight_dict_path")
        self.iteration_dir = self.config.get("iteration_dir")
        self.requirements_paths = self.config.get("requirements_paths")
        
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
    
    @public_method
    def load_model(self):
        """
        Load a model defined in the model_specs.
        """
        # Load the create_model function for the iteration
        create_model = self._import_function(self.model_modules_path, self.model_function)
        
        # Pass the model's parameters from the config dict
        if callable(create_model):
            model = create_model(**self.model_args)
        else:
            raise TypeError("create_model is not callable")
            
        return model
    
    @public_method
    def compile_model(self, model):
        """
        Compile the model based on its specifications.
        
        Args:
            model (tf.keras.Model): The TensorFlow model to compile.
            model_name (str): Name of the model for retrieving its specifications.
            
        Returns:
            tf.keras.Model: The compiled TensorFlow model.
        """
        # Retrieve model specifications
        optimizer_config = self.optimizer
        
        # Dynamically create the optimizer
        if isinstance(optimizer_config, dict):
            optimizer_type = optimizer_config.pop("type", "adam")  # Default to Adam if type is not specified
            optimizer_class = getattr(tf.keras.optimizers, optimizer_type.capitalize(), None)
            
            if optimizer_class is None:
                raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized.")
            
            # Instantiate the optimizer with remaining parameters in optimizer_config
            optimizer = optimizer_class(**optimizer_config)
        else:
            # If `optimizer_config` is not a dictionary, assume it's a valid TensorFlow optimizer
            optimizer = tf.keras.optimizers.get(optimizer_config)
            
        # Determine the loss function:
        # If custom_loss is provided (non-empty dict), import and use that custom loss.
        if self.custom_loss and isinstance(self.custom_loss, dict) and len(self.custom_loss) > 0:
            # Use self.model_modules_path for the module path and "custom_loss" as the function name.
            path = self.custom_loss["custom_loss_path"]
            module_name = self.custom_loss["module_name"]
            loss_fn = self._import_function(path, module_name)
            print("Using custom loss function.")
        else:
            loss_fn = self.loss
            print("Using standard loss function:", self.loss)
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=self.metrics
        )
        
        print("Successfully compiled model...")
        
        return model
        
    def _load_feature_metadata(self, feature_category):
        """
        Load feature metadata from the corresponding JSON file dynamically.
        - Searches for files matching the expected pattern.
        - If exact match is missing, searches for similar files.
        """
        # Define expected JSON filename pattern
        expected_filename = f"{feature_category}_feature_description.json"
        
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
    
        # Check for exact match
        json_file_path = os.path.join(self.data_dir, expected_filename)
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
        
    def _load_target_metadata(self):
        """
        Load target metadata from the corresponding JSON file.
        """
        json_file = os.path.join(self.data_dir, "targets_feature_description.json")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Target description JSON not found: {json_file}")

        with open(json_file, "r") as f:
            metadata = json.load(f)

        return metadata

    def _parse_tfrecord_fn(self, example_proto, feature_metadata, category):
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
        category_config = self.feature_categories[category]
        if category_config["reshape"]:
            feature_tensor = tf.reshape(feature_tensor, category_config["shape"])
    
        return feature_tensor
    
    @tf.autograph.experimental.do_not_convert
    def _load_tfrecord_dataset(self, feature_files, target_files):
        """
        Load and process datasets from multiple aligned feature and target TFRecord files.
        Ensures correct structure for model input.
        """
        feature_metadata = {category: self._load_feature_metadata(category) for category in self.feature_categories}
    
        feature_datasets = {
            category: tf.data.TFRecordDataset(files).map(
                lambda x: self._parse_tfrecord_fn(x, feature_metadata[category], category),
                num_parallel_calls=tf.data.AUTOTUNE
            ) for category, files in feature_files.items()
        }
    
        # Load and parse target dataset separately (only once)
        target_dataset = tf.data.TFRecordDataset(target_files).map(
            lambda x: tf.io.parse_single_example(
                x, {'target': tf.io.FixedLenFeature([], tf.int64)}
            )['target'], num_parallel_calls=tf.data.AUTOTUNE
        )
    
        # Convert feature dictionary to tuple (preserves ordering)
        feature_tuple_dataset = tf.data.Dataset.zip((
            tuple(feature_datasets.values()),  # Tuple of feature inputs
            target_dataset                     # Single target label
        ))
    
        feature_tuple_dataset = feature_tuple_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return feature_tuple_dataset

    def _match_feature_target_files(self, feature_category, dataset_type):
        """
        Match feature and target files based on numbering for proper alignment.
        """
        # List all files
        tfrecord_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')])

        # Filter for feature and target files of the correct category and dataset type (train/val/test)
        feature_files = sorted([os.path.join(self.data_dir, f) for f in tfrecord_files 
                                if f.startswith(feature_category) and dataset_type in f])
        target_files = sorted([os.path.join(self.data_dir, f) for f in tfrecord_files 
                               if f.startswith("targets") and dataset_type in f])

        # Ensure the number of feature and target files match
        if len(feature_files) != len(target_files):
            raise ValueError(f"Mismatch between feature ({len(feature_files)}) and target ({len(target_files)}) files for {dataset_type}.")

        return feature_files, target_files

    def _group_files_by_number(self, feature_category, dataset_type):
        """
        Groups feature and target files by their number (0,1,2,...) for training order consistency.
        """
        tfrecord_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')])

        feature_files = [f for f in tfrecord_files if f.startswith(feature_category) and dataset_type in f]
        target_files = [f for f in tfrecord_files if "targets" in f and dataset_type in f]

        file_dict = {}

        for f in feature_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num not in file_dict:
                    file_dict[num] = {'features': [], 'targets': []}
                file_dict[num]['features'].append(os.path.join(self.data_dir, f))

        for f in target_files:
            match = re.search(r'_(\d+)_', f)
            if match:
                num = int(match.group(1))
                if num in file_dict:  # Ensuring targets align with features
                    file_dict[num]['targets'].append(os.path.join(self.data_dir, f))

        return file_dict  # Returns {number: {features: [...], targets: [...]}}
    
    def _save_model_and_weights(self, model, model_name, save_dir):
        """
        Save the entire model and its weights separately.
    
        Args:
            model (tf.keras.Model): The trained model.
            model_name (str): Name of the model.
            save_dir (str): Directory path where the model and weights should be saved.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        # Define file paths
        model_path = os.path.join(save_dir, f"{model_name}.keras")
        weights_path = os.path.join(save_dir, f"{model_name}.weights.h5")
    
        # Save the full model
        model.save(model_path)
        print(f"Full model saved to: {model_path}")
    
        # Save only the weights
        model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
        
    def _copy_file(self, source_path, export_path):
        """
        Copies a file from source_path to export_path.
        
        If export_path is a directory, the file is copied into that directory with its original filename.
        If export_path is a full file path, the file is copied to that location.
        
        Args:
            source_path (str): The path of the file to copy.
            export_path (str): The destination directory or file path where the file should be saved.
        
        Returns:
            None
        """
        # Check if the source file exists.
        if not os.path.isfile(source_path):
            print(f"Source file does not exist: {source_path}")
            return
    
        # Determine the final destination path.
        if os.path.isdir(export_path):
            # If export_path is a directory, append the source filename.
            destination_path = os.path.join(export_path, os.path.basename(source_path))
        else:
            # If export_path is a full file path, use it directly.
            destination_path = export_path
            # Ensure the destination directory exists.
            dest_dir = os.path.dirname(destination_path)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
        
        try:
            shutil.copy2(source_path, destination_path)
            print(f"File successfully copied from {source_path} to {destination_path}")
        except Exception as e:
            print(f"Error copying file: {e}")

    @public_method
    def fit_model(self, model):
        """
        Iteratively loads aligned feature and target TFRecord files in batches,
        ensuring each epoch processes all aligned sets (0,1,2,...) before validation.
        Supports dynamically specified feature categories.
        """
        # Create the unique save directory name
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        save_dir = os.path.join(self.iteration_dir, dt_string)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Successfully created the save directory:", save_dir)
            
        # Copy the config.py and file with model architecture information
        self._copy_file(self.requirements_paths["config"], save_dir)
        self._copy_file(self.requirements_paths["model"], save_dir)
        
        # Attempt to load class weights if the file exists
        if os.path.exists(self.weight_dict_path):
            with open(self.weight_dict_path, 'r') as f:
                class_weights = json.load(f)
            # Convert keys from strings to ints
            class_weights = {int(k): v for k, v in class_weights.items()}
            print("Successfully loaded class weights.")
        
        tfrecord_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')])
    
        # Ensure all specified feature categories exist
        required_categories = set(self.feature_categories.keys())
        existing_categories = set(f.split('_')[0] for f in tfrecord_files if 'features' in f)
        missing_categories = required_categories - existing_categories
        if missing_categories:
            raise ValueError(f"Missing required feature categories: {missing_categories}")
    
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Initialize accumulators for training metrics.
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            num_train_batches = 0
    
            # Training Phase (ensure all feature sets are loaded together)
            train_groups = {category: self._group_files_by_number(category, 'train') 
                            for category in self.feature_categories}
    
            # Loop through all numbered datasets (0, 1, 2, etc.)
            for num in sorted(train_groups[next(iter(self.feature_categories))].keys()):
                feature_files = {
                    category: train_groups[category].get(num, {}).get('features', []) 
                    for category in self.feature_categories}
                target_files = train_groups[
                    next(iter(self.feature_categories))].get(num, {}).get('targets', [])
    
                # Ensure all feature categories are present
                if any(not files for files in feature_files.values()) or not target_files:
                    continue  # Skip if any feature category is missing
    
                print(f"Training on set {num}")
    
                # Load dataset with all feature sets
                train_dataset = self._load_tfrecord_dataset(feature_files, target_files)
    
                # Train model with all feature inputs; include class_weight if available
                if class_weights is not None:
                    history = model.fit(train_dataset, epochs=1, verbose=1, class_weight=class_weights)
                else:
                    history = model.fit(train_dataset, epochs=1, verbose=1)
                    
                # Assuming 'loss' and 'accuracy' are tracked metrics
                # (Update the keys if your metric names differ.)
                batch_loss = history.history.get('loss', [0])[-1]
                batch_acc = history.history.get('accuracy', [0])[-1]
                train_loss_sum += batch_loss
                train_acc_sum += batch_acc
                num_train_batches += 1
    
                del train_dataset
                gc.collect()
                
            # Compute average training metrics for this epoch
            avg_train_loss = train_loss_sum / num_train_batches if num_train_batches else 0
            avg_train_acc = train_acc_sum / num_train_batches if num_train_batches else 0
    
            # Initialize accumulators for validation metrics.
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            num_val_batches = 0
            
            # We also want to accumulate per-sample predictions for the CSV.
            pred_rows = []  # Each element: [epoch, predicted_category, actual_target, prediction_confidence]
                
            # Validation Phase
            val_groups = {category: self._group_files_by_number(category, 'val') 
                          for category in self.feature_categories}
    
            for num in sorted(val_groups[next(iter(self.feature_categories))].keys()):
                feature_files = {category: val_groups[category].get(num, {}).get('features', []) 
                                 for category in self.feature_categories}
                target_files = val_groups[next(iter(self.feature_categories))].get(num, {}).get('targets', [])
    
                if any(not files for files in feature_files.values()) or not target_files:
                    continue  
    
                print(f"Validating on set {num}")
    
                val_dataset = self._load_tfrecord_dataset(feature_files, target_files)
                results = model.evaluate(val_dataset, verbose=1)
                # Typically, results[0] is loss and results[1] is accuracy.
                val_loss_sum += results[0]
                if len(results) > 1:
                    val_acc_sum += results[1]
                num_val_batches += 1
                
                # Now, run predictions on this validation chunk.
                # Since our dataset is a tf.data.Dataset of (features, target),
                # we use model.predict to get the output probabilities.
                predictions = model.predict(val_dataset)
                # Collect actual targets from the dataset.
                # (We assume that the dataset yields (features, target) tuples.)
                actual_targets = []
                for _, t in val_dataset:
                    # Flatten in case targets come with extra dimensions.
                    actual_targets.extend(t.numpy().flatten().tolist())
                # Ensure predictions is a NumPy array.
                predictions = np.array(predictions)
                # For each sample, the predicted category is the argmax and the confidence is the max probability.
                predicted_categories = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
                # Add one row per sample.
                for pred, actual, conf in zip(predicted_categories, actual_targets, confidences):
                    pred_rows.append([epoch + 1, pred, actual, conf])
    
                del val_dataset
                gc.collect()
                
            # Compute average validation metrics for this epoch
            avg_val_loss = val_loss_sum / num_val_batches if num_val_batches else 0
            avg_val_acc = val_acc_sum / num_val_batches if num_val_batches else 0
    
            # Construct a filename that includes these metrics.
            # For example: "epoch1_trainLoss_0.1234_trainAcc_0.9876_valLoss_0.2345_valAcc_0.9765.h5"
            model_filename = (
                f"epoch{epoch+1}_"
                f"trainLoss_{avg_train_loss:.4f}_"
                f"trainAcc_{avg_train_acc:.4f}_"
                f"valLoss_{avg_val_loss:.4f}_"
                f"valAcc_{avg_val_acc:.4f}.h5"
            )
            model_path = os.path.join(save_dir, model_filename)
            model.save(model_path)
            model.save_weights(model_path)
            print(f"Saved model for epoch {epoch+1} at: {model_path}")
            
            # Append per-sample predictions to the predictions CSV.
            predictions_csv_path = os.path.join(save_dir, f"{epoch}_predictions.csv")
            with open(predictions_csv_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header row for sample-level predictions
                writer.writerow(["epoch", "predicted_category", "actual_target", "prediction_confidence"])
                for row in pred_rows:
                    writer.writerow(row)
            print(f"Appended predictions for epoch {epoch+1} to CSV file: {predictions_csv_path}")
            
    @public_method
    def train_models(self):
        """
        Train all models defined in model_specs.

        Returns:
            dict: Training histories for each model.
        """
        model = self.load_model()
        model = self.compile_model(model)
        self.fit_model(model)
    