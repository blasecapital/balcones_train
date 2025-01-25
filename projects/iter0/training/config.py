# config.py


config = {
    # Arguements for load_training_data
    # Provide the database path and the query as a set for each feature file
    "source_query": [("FEATURE_DATABASE", 
                      """
                      SELECT 
                        hf.*, t.target, t.hours_passed, t.buy_sl_time, t.sell_sl_time
                      FROM 
                        hourly_features hf
                      INNER JOIN 
                        targets t
                      ON 
                        hf.pair = t.pair AND hf.date = t.date
                      WHERE 
                        hf.pair = 'AUDUSD' AND hf.date > '2023-06-01';
                      """,
                      "model1")
                      ],
    
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