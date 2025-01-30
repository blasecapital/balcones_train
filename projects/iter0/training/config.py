# config.py


config = {
    # Arguements for load_training_data
    # Name the data key (name of feature set or target set), write the .env
    # reference name for the source db_path and the associated query
    "source_query": {
        "iter0_hourly_features" : ("FEATURE_DATABASE", 
         """
         SELECT * FROM hourly_features
         WHERE pair IN ('AUDUSD', 'EURUSD')
         AND date > '2023-10-01'
         ORDER BY pair, date
         """),
         "iter0_targets" : ("TARGET_DATABASE",
         """
         SELECT * FROM targets
         WHERE pair = 'AUDUSD'
         """)},
         
    "project_directory": (r'C:\Users\brand\OneDrive\Blase Capital Mgmt'
                          r'\deep_learning\projects\iter0\training'),
    
    # Use accross modules
    "primary_key": ['date', 'pair'],
    
    # Args for raw data processing
    # Path to low level functionality to pass to ProcessRawData
    "data_processing_modules_path": (r'C:\Users\brand\OneDrive'
                                     r'\Blase Capital Mgmt\deep_learning'
                                     r'\projects\iter0\training'
                                     r'\iter0_process_raw_data.py'),
    "filter_function": 'filter_indices',
    # (name of function, dataframe to apply to (key of df_dict))
    "feature_engineering": [('feature_engineering', 'model1')],
    "target_engineering": [('target_engineering', 'model1')],
    "define_features": [('df_features', 'model1')],
    # (target column name, associated dataframe)
    "define_targets": [('target', 'model1')],
    # (dict mapping categories to their codes, associated dataframe)
    "category_index": [({'loss': 0, 'buy': 1, 'sell': 2, 'wait': 0}, 'model1')],
    "scaler_save_path": (r'C:\Users\brand\OneDrive'
                         r'\Blase Capital Mgmt\deep_learning'
                         r'\projects\iter0\training'),
    # (samples, timesteps, features, dataframe to apply to (df_key))
    # Example (-1, 48, 4, 'model1')
    "reshape": [(-1, 48, 4, 'model1')],
    
    # Args for training
    "model_specs": {
        "model1": {
            "model_modules_path": (r'C:\Users\brand\OneDrive'
                                   r'\Blase Capital Mgmt\deep_learning'
                                   r'\projects\iter0\training'
                                   r'\iter0_training.py'),
            "initial_bias_path": (),
            "model_save_path": (r'C:\Users\brand\OneDrive'
                                r'\Blase Capital Mgmt\deep_learning'
                                r'\projects\iter0\training'),
            "save_requirements_path": (r'C:\Users\brand\OneDrive'
                                       r'\Blase Capital Mgmt\deep_learning'
                                       r'\projects\iter0\training'),
            "model_function": 'create_model',
            "custom_loss": {},
            "loss": {"output_layer": "categorical_crossentropy"},
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