# iter0_training.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, SpatialDropout1D


def create_model(
        timesteps=None,
        features=None,
        output_dim=1,
        lstm_layers=[32, 16],
        output_activation="softmax",
        dropout_rate=0.5,
        batch_size=None):
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
    # Input layer
    inputs = Input(shape=(timesteps, features), batch_size=batch_size, name="input_layer")
    
    # Add LSTM layers
    x = inputs
    for i, units in enumerate(lstm_layers):
        return_sequences = (i < len(lstm_layers) - 1)  # Return sequences for all but the last LSTM
        x = LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_layer_{i+1}",
            dropout=dropout_rate
        )(x)
    
    # Output layer
    outputs = Dense(output_dim, activation=output_activation, name="output_layer")(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name="stacked_lstm_model")
    model.summary()
    return model