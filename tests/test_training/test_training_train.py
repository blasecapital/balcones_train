# test_training_train.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2


def create_model(
        input_dim, 
        output_dim=1, 
        hidden_layers=[32, 16], 
        activation="relu", 
        output_activation="sigmoid",
        l2_reg=0.01,
        dropout_rate=0.5):
    """
    Create a dense-layer TensorFlow neural network model with L2 regularization and dropout.
    
    Args:
        input_dim (int): Number of features in the input data.
        output_dim (int): Number of targets in the output data.
        hidden_layers (list): List specifying the number of units in each hidden layer.
        activation (str): Activation function for hidden layers (default: "relu").
        output_activation (str or None): Activation function for the output layer.
            Use "sigmoid" for binary classification, "softmax" for multi-class, or
            None for regression (default: None).
        l2_reg (float): L2 regularization factor (default: 0.01).
            Set to 0 for no regularization.
        dropout_rate (float): Fraction of units to drop (default: 0.5). 
            Set to 0 to disable dropout.

    Returns:
        tf.keras.Model: The compiled TensorFlow model.
    """
    # Input layer
    inputs = Input(shape=(input_dim,), name="input_layer")
    
    # Hidden layers with L2 regularization and Dropout
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = Dense(
            units, 
            activation=activation, 
            kernel_regularizer=l2(l2_reg),  # Add L2 regularization
            name=f"hidden_layer_{i+1}"
        )(x)
        
        # Add dropout after each hidden layer
        if dropout_rate > 0:
            x = Dropout(rate=dropout_rate, name=f"dropout_{i+1}")(x)
    
    # Output layer
    outputs = Dense(
        output_dim, 
        activation=output_activation, 
        kernel_regularizer=l2(l2_reg),  # Optional: Add L2 regularization
        name="output_layer"
    )(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="dense_nn")
    
    # Display architecture
    model.summary()
    
    # Return the uncompiled model
    return model
