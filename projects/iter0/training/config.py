# config.py


config = {
    # Arguements for load_training_data
    # Provide the database path and the query as a set for each feature file
    "source_query": [("FEATURE_DATABASE", 
                      """ 
                      SELECT * FROM hourly_features
                      WHERE pair = 'AUDUSD'
                      AND date > '2023-06-01'
                      """,
                      "model1")
                      ],
    
    # Use accross modules
    "primary_key": ['date', 'pair'],
    
    # Args for raw data processing
    # Path to low level functionality to pass to ProcessRawData
    "data_processing_modules_path": (r'C:\Users\BrandonBlase\balcones_system'
                                     r'\model_development\tests\test_training'
                                     r'\test_training_process_raw_data.py'),
    "filter_function": 'filter_indices',
    # (name of function, dataframe to apply to (key of df_dict))
    "feature_engineering": [('feature_engineering', 'model1')],
    "target_engineering": [('target_engineering', 'model1')],
    "define_features": [('df_features', 'model1')],
    # (target column name, associated dataframe)
    "define_targets": [('target', 'model1')],
    # (dict mapping categories to their codes, associated dataframe)
    "category_index": [({'buy': 0, 'sell': 1}, 'model1')],
    "scaler_save_path": (r'C:\Users\BrandonBlase\balcones_system'
                         r'\model_development\tests\test_training'),
    # (samples, timesteps, features, dataframe to apply to (df_key))
    # Example (-1, len(feature_columns_hourly) // 4, 4)
    "reshape": [],
    
    # Args for training
    "model_specs": {
        "model1": {
            "model_modules_path": (r'C:\Users\BrandonBlase\balcones_system'
                                   r'\model_development\tests\test_training'
                                   r'\test_training_train.py'),
            "initial_bias_path": (),
            "model_save_path": (r'C:\Users\BrandonBlase\balcones_system'
                                r'\model_development\tests\test_training'),
            "save_requirements_path": (r'C:\Users\BrandonBlase\balcones_system'
                                       r'\model_development\tests\test_training'),
            "model_function": 'create_model',
            "custom_loss": {},
            "loss": {"output_layer": "binary_crossentropy"},
            "metrics": ["accuracy"],
            "learning_rate_schedule": [],
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "epochs": 10,
            "batch_size": 32,
            "early_stopping": '',
            "checkpoint": ''
            }
        }
    ,
    
    }