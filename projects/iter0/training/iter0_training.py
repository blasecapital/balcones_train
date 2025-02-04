# iter0_training.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate
from tensorflow.keras.regularizers import l2


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
    x_hourly = LSTM(256, return_sequences=True, kernel_regularizer=l2(l2_strength))(hourly_input)
    x_hourly = Dropout(dropout_rate)(x_hourly)
    x_hourly = LSTM(128, kernel_regularizer=l2(l2_strength))(x_hourly)
    
    # Engineered Layers
    eng_input = Input(shape=(n_dense_features,))
    x_eng = Dense(64, activation=activation, kernel_regularizer=l2(l2_strength))(eng_input)
    
    # Concatenate Layers
    concatenated = Concatenate()([x_hourly, x_eng])
    x = Dense(256, activation=activation)(concatenated)
    
    # Output layer
    output = Dense(n_targets, activation=output_activation, name="output_layer")(x)

    # Define the model
    model = Model(inputs=[hourly_input, eng_input], outputs=output)
    model.summary()
    return model