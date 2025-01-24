# train.py


import importlib.util
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import os
# Suppress TensorFlow INFO logs and disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from config import config
from utils import public_method


class Train:
    def __init__(self, training_data: dict):
        # The initialized object
        self.training_data = training_data
        
        # Initialize config attributes
        self.model_specs = config.get("model_specs")
        
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
        
    def _load_model(self, model_name):
        """
        Load a model defined in the model_specs.
        """
        input_dim = self.training_data['features'][model_name]['train'].shape[1]
        
        spec = self.model_specs[model_name]
        model_function = self._import_function(spec["model_modules_path"], spec["model_function"])
        return model_function(input_dim)
    
    def _compile_model(self, model, model_name):
        """
        Compile the model based on its specifications.
        
        Args:
            model (tf.keras.Model): The TensorFlow model to compile.
            model_name (str): Name of the model for retrieving its specifications.
            
        Returns:
            tf.keras.Model: The compiled TensorFlow model.
        """
        # Retrieve model specifications
        spec = self.model_specs[model_name]
        optimizer_config = spec["optimizer"]
        
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
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=spec["loss"],
            metrics=spec["metrics"]
        )
        
        print("Successfully compiled model...")
        
        return model
    
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
        
    def _weights_dict(self, model_name):
        """
        Compute class weights for imbalanced data.
        """
        y_integers = self.training_data['targets'][model_name]['train']
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_integers), 
            y=y_integers
        )
        return dict(enumerate(class_weights))
        
        
    def _fit_model(self, model, model_name):
        """
        Fit the model using either TensorFlow's built-in `fit` method or a custom loop.

        Args:
            model (tf.keras.Model): The compiled TensorFlow model.
            model_name (str): Name of the model being trained.
        
        Returns:
            tf.keras.callbacks.History: Training history.
        """
        spec = self.model_specs[model_name]
        train_data = self.training_data['features'][model_name]['train']
        val_data = self.training_data['features'][model_name]['val']
        train_targets = self.training_data['targets'][model_name]['train']
        val_targets = self.training_data['targets'][model_name]['val']

        callbacks = []
        if spec["early_stopping"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(**spec["early_stopping"]))
        if spec["checkpoint"]:
            checkpoint_path = spec["checkpoint"]
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True))

        history = model.fit(
            x=train_data,
            y=train_targets,
            validation_data=(val_data, val_targets),
            epochs=spec["epochs"],
            batch_size=spec["batch_size"],
            class_weight=self._weights_dict(model_name),
            callbacks=callbacks
        )

        # Save the trained model and weights
        model_save_path = spec["model_save_path"]
        self._save_model_and_weights(model, model_name, model_save_path)
        print(f"Model '{model_name}' saved to: {model_save_path}")
        
        return history
    
    def _custom_fit(model, train_data, val_data, optimizer, loss_fn, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training step
            for x_batch, y_batch in train_data:
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    loss = loss_fn(y_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Validation step
            for x_batch, y_batch in val_data:
                predictions = model(x_batch, training=False)
                val_loss = loss_fn(y_batch, predictions)
            
            print(f"Epoch {epoch + 1}: Loss: {loss.numpy()}, Val Loss: {val_loss.numpy()}")
            
    def _collect_preprocessing_config(self):
        """
        Collect preprocessing configuration attributes from the training_data dictionary.

        Returns:
            dict: A dictionary containing the preprocessing configuration.
        """
        process_data_config = self.training_data.get("process_data_config", {})
        return process_data_config
            
    def _save_requirements(self, model_name):
        """
        Saves the imported `config` dictionary and the `process_data_config` to a new `.py` 
        file in the specified `save_config_path` directory. 
    
        Args:
            model_name (str): The name of the model for which the requirements are being saved.
        
        Raises:
            FileExistsError: If the `.py` file already contains a `config` dictionary.
            Exception: If the `save_config_path` is invalid or inaccessible.
        """
        # Validate the save path
        save_config_dir = self.model_specs[model_name]['save_requirements_path']
        if not os.path.isdir(save_config_dir):
            raise Exception(f"The specified path {save_config_dir} is not a valid directory.")
    
        # Construct the file path for the config file
        save_file_path = os.path.join(save_config_dir, 'config.py')
                        
        # Collect the process_data_config
        process_data_config = self._collect_preprocessing_config()
        
        with open(save_file_path, 'w') as file:
            file.write("# Auto-generated configuration file\n\n")
            file.write("# Main configuration\n")
            file.write("config = ")
            file.write(repr(config))  # Write the dictionary as a string
            file.write("\n\n")
            
            file.write("# Preprocessing configuration\n")
            file.write("process_data_config = ")
            file.write(repr(process_data_config))  # Write the `process_data_config` dictionary as a string
        
        print(f"Configuration saved successfully to {save_file_path}")
        
    @public_method
    def train_models(self):
        """
        Train all models defined in model_specs.

        Returns:
            dict: Training histories for each model.
        """
        histories = {}
        for model_name in self.model_specs:
            print(f"Starting training for model '{model_name}'...") 
            self._save_requirements(model_name)
            model = self._load_model(model_name)
            model = self._compile_model(model, model_name)
            history = self._fit_model(model, model_name)
            histories[model_name] = history
            print(f"Completed training for model '{model_name}'.")
        return histories
    