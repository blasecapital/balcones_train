# iter0_training.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate
from tensorflow.keras.regularizers import l2
import tensorflow as tf


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


def create_model(n_hours_back_hourly, n_ohlc_features, l2_strength, dropout_rate,
                 n_dense_features, activation, n_targets, output_activation):
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
    x_eng = Dense(32, activation=activation, kernel_regularizer=l2(l2_strength))(eng_input)
    
    # Concatenate Layers
    concatenated = Concatenate()([x_hourly, x_eng])
    x = Dense(512, activation=activation)(concatenated)
    x = Dense(256, activation=activation)(x)
    x = Dense(128, activation=activation)(x)
    
    # Output layer
    output = Dense(n_targets, activation=output_activation, name="output_layer")(x)

    # Define the model
    model = Model(inputs=[hourly_input, eng_input], outputs=output)
    model.summary()
    return model