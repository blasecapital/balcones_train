# iter0_training.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM


def create_model(
        timesteps=None,
        features=None,
        output_dim=1, 
        hidden_layers=[32, 16], 
        activation="relu", 
        output_activation="softmax",
        l2_reg=0.01,
        dropout_rate=0.5):
    """
    Create a model with LSTM layers for time-series data.
    """
    inputs = Input(shape=(timesteps, features), name="input_layer")

    # Add LSTM layer
    x = LSTM(32, activation=activation, recurrent_activation=output_activation)(inputs)
    
    # Add output layer
    outputs = Dense(1, activation="softmax")(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model