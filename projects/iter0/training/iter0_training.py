# iter0_training.py


import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, GlobalAveragePooling1D, LayerNormalization, 
    MultiHeadAttention, Add, Flatten, TimeDistributed, LSTM)
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np


def custom_loss(y_true, y_pred, max_loss=5):
    minority_class_indices=[1,2]
    penalty_weight = 0.005
    
    # Ensure y_pred is float32
    y_pred = tf.cast(y_pred, tf.float32)

    # Avoid log(0) by adding a small constant (epsilon)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Calculate categorical cross-entropy loss
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Clip the categorical cross-entropy loss to avoid extreme values
    ce_loss = tf.clip_by_value(ce_loss, 0.00001, max_loss)

    # Identify predicted and true classes
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)

    # Identify true positives and false positives for the minority class
    is_minority_class_pred = tf.reduce_any(
        tf.equal(tf.expand_dims(y_pred_classes, axis=-1), minority_class_indices), axis=-1)
    is_minority_class_true = tf.reduce_any(
        tf.equal(tf.expand_dims(y_true_classes, axis=-1), minority_class_indices), axis=-1)

    true_positives = tf.logical_and(tf.equal(y_true_classes, y_pred_classes), is_minority_class_true)
    false_positives = tf.logical_and(tf.not_equal(y_true_classes, y_pred_classes), is_minority_class_pred)

    # Calculate precision for the minority class
    true_positive_count = tf.reduce_sum(tf.cast(true_positives, tf.float32))
    false_positive_count = tf.reduce_sum(tf.cast(false_positives, tf.float32))

    # Ensure no division by zero
    precision = tf.where(
        tf.equal(true_positive_count + false_positive_count, 0),
        0.0,
        true_positive_count / (true_positive_count + false_positive_count + epsilon)
    )

    # Compute the penalty based on precision
    precision_penalty = (1.0 - precision) * penalty_weight

    # Apply the penalty only to false positives of the minority class
    precision_penalty_per_sample = tf.where(false_positives, precision_penalty, 0.0)

    # Clip the precision penalty to avoid extreme values
    precision_penalty_per_sample = tf.clip_by_value(precision_penalty_per_sample, 0, max_loss)

    # Calculate the total loss per sample
    total_loss_per_sample = ce_loss + precision_penalty_per_sample

    # Ensure numerical stability in the total loss
    total_loss_per_sample = tf.clip_by_value(total_loss_per_sample, 0, max_loss)

    return total_loss_per_sample


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=2):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.best = np.Inf
        self.wait = 0

    def on_aggregated_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            print(f"[EarlyStopping] '{self.monitor}' not found in logs.")
            return
        if current < self.best:
            self.best = current
            self.wait = 0
            print(f"[EarlyStopping] Epoch {epoch+1}: {self.monitor} improved to {current:.4f}.")
        else:
            self.wait += 1
            print(f"[EarlyStopping] Epoch {epoch+1}: {self.monitor} did not improve (wait {self.wait}/{self.patience}).")
            if self.wait >= self.patience:
                print(f"[EarlyStopping] Early stopping triggered at epoch {epoch+1}.")
                self.model.stop_training = True


class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = np.Inf
        self.wait = 0

    def on_aggregated_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            print(f"[ReduceLROnPlateau] '{self.monitor}' not found in logs.")
            return
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = max(old_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"[ReduceLROnPlateau] Epoch {epoch+1}: LR reduced from {old_lr:.6f} to {new_lr:.6f}.")
                self.wait = 0


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = np.Inf

    def on_aggregated_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            print(f"[ModelCheckpoint] '{self.monitor}' not found in logs.")
            return
        if current < self.best:
            self.best = current
            self.model.save_weights(self.filepath)
            print(f"[ModelCheckpoint] Epoch {epoch+1}: Checkpoint saved to {self.filepath} with {self.monitor} = {current:.4f}.")

# --------------------------
# AggregateCallbacks: container for all aggregated callbacks
# --------------------------

class AggregateCallbacks:
    def __init__(self, monitor_metric='val_loss', patience=2, save_dir=None,
                 use_early_stopping=True, use_reduce_lr=True, use_model_checkpoint=True,
                 use_tensorboard=True, use_csv_logger=True):
        """
        Instantiate only the callbacks you want to use. If a particular callback is not desired,
        set its corresponding use_* flag to False.
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.use_tensorboard = use_tensorboard

        # EarlyStopping
        self.early_stopping = CustomEarlyStopping(monitor=monitor_metric, patience=patience) if use_early_stopping else None
        
        # ReduceLROnPlateau
        self.reduce_lr = CustomReduceLROnPlateau(monitor=monitor_metric, factor=0.8, patience=3, min_lr=1e-6) if use_reduce_lr else None
        
        # ModelCheckpoint (requires a valid save_dir)
        if use_model_checkpoint:
            if save_dir is None:
                raise ValueError("save_dir must be provided for model checkpointing.")
            checkpoint_filepath = os.path.join(save_dir, "best_model.h5")
            self.model_checkpoint = CustomModelCheckpoint(filepath=checkpoint_filepath, monitor=monitor_metric)
        else:
            self.model_checkpoint = None

        # CSV logging: write aggregated metrics to a CSV file.
        if use_csv_logger:
            if save_dir is None:
                raise ValueError("save_dir must be provided for CSV logging.")
            self.csv_log_path = os.path.join(save_dir, "aggregated_training_log.csv")
            # Write header if file doesn't exist.
            if not os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, 'w') as f:
                    f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")
        else:
            self.csv_log_path = None

    def on_aggregated_epoch_end(self, epoch, logs, model):
        """
        This method should be called at the end of an outer epoch with aggregated metrics.
        It will invoke each aggregated callback that was configured.
        """
        # Make sure the model is assigned to each callback that uses it.
        if self.early_stopping is not None:
            self.early_stopping.model = model
            self.early_stopping.on_aggregated_epoch_end(epoch, logs)
        if self.reduce_lr is not None:
            self.reduce_lr.model = model
            self.reduce_lr.on_aggregated_epoch_end(epoch, logs)
        if self.model_checkpoint is not None:
            self.model_checkpoint.model = model
            self.model_checkpoint.on_aggregated_epoch_end(epoch, logs)
        if self.csv_log_path is not None:
            import csv
            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, logs.get('train_loss', 0),
                                 logs.get('train_accuracy', 0),
                                 logs.get('val_loss', 0),
                                 logs.get('val_accuracy', 0)])
            print(f"[CSVLogger] Epoch {epoch+1}: Metrics appended to {self.csv_log_path}.")
    

def create_model(n_hours_back_hourly, n_ohlc_features, l2_strength, dropout_rate,
                 n_dense_features, activation, n_targets, output_activation,
                 initial_bias):
    """
    Create a TensorFlow model with multiple stacked LSTM layers optimized for GPU (cuDNN) support.

    Args:
        timesteps (int): Number of time steps in the input sequence.
        features (int): Number of features for each time step.
        output_dim (int): Number of targets in the output data.
        lstm_layers (list): List specifying the number of units in each LSTM layer.
        output_activation (str or None): Activation function for the output layer.
            Use "sigmoid" for binary classification, "softmax" for multi-class, or
            None for regression (default: "softmax").
        dropout_rate (float): Fraction of units to drop (default: 0.5). 
            Set to 0 to disable dropout.
        batch_size (int or None): Batch size for fixed input size (default: None).

    Returns:
        tf.keras.Model: The compiled TensorFlow model.
    """
    # Hourly LSTM Layers
    hourly_input = Input(shape=(n_hours_back_hourly, n_ohlc_features))
    x_hourly = LSTM(512, return_sequences=True, kernel_regularizer=l2(l2_strength))(hourly_input)
    x_hourly = Dropout(dropout_rate)(x_hourly)
    x_hourly = LSTM(512, return_sequences=True, kernel_regularizer=l2(l2_strength))(x_hourly)
    x_hourly = LSTM(512, kernel_regularizer=l2(l2_strength))(x_hourly)
    
    # Engineered Layers
    eng_input = Input(shape=(n_dense_features,))
    x_eng = Dense(64, activation=activation, kernel_regularizer=l2(l2_strength))(eng_input)
    
    # Concatenate Layers
    concatenated = Concatenate()([x_hourly, x_eng])
    x = Dense(512, activation=activation)(concatenated)
    x = Dense(512, activation=activation)(x)
    
    # Output layer
    output = Dense(n_targets, 
                   activation=output_activation, 
                   bias_initializer=tf.keras.initializers.Constant(initial_bias),
                   name="output_layer")(x)

    # Define the model
    model = Model(inputs=[hourly_input, eng_input], outputs=output)
    model.summary()
    return model