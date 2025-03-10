# config.py


config = {
    # Arguements for load_training_data
    # Name the data key (name of feature set or target set), write the .env
    # reference name for the source db_path and the associated query
    "source_query": {
        "iter0_hourly_features" : ("CLEAN_FEATURE_DATABASE", 
         """
         SELECT * FROM hourly_features
         """),
         "iter0_targets" : ("TARGET_DATABASE",
         """
         SELECT * FROM targets
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
    # Dictionary for collecting bad data primary keys, the dict key is a 
    # metadata name and its item is a set containing the env key to the 
    # database, a query for specifying the table and data to filter, and 
    # the name of the filter function
    "clean_functions": {
        "iter0_hourly_features": ("FEATURE_DATABASE", 
         """
         SELECT * FROM hourly_features
         """,
         'filter_hourly'),
        "iter0_targets": ("TARGET_DATABASE",
        """
        SELECT * FROM targets
        WHERE pair = 'AUDUSD'
        """,
        'filter_targets')
        },
    "bad_keys_path": (r'C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning'
                      r'\projects\iter0\training\bad_keys'),
    
    # Dictionary for cleaning and saving
    "align": {
        "iter0_hourly_features": ("FEATURE_DATABASE",
            """
            SELECT * FROM hourly_features
            """
            ),
        "iter0_targets": ("TARGET_DATABASE",
        """
        SELECT * FROM targets
        """)
        },
    
    
    # Dicts for PrepData engineer function
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) Query
    #       3) function name
    #       4) save table name
    "feature_engineering": {
        "iter0_feature_engineer": (
            "CLEAN_FEATURE_DATABASE",
            """
            SELECT date, pair FROM hourly_features
            """,
            'feature_engineering',
            'dense_features'
            )
        },
    # Only use if creating new targets in a new table
    # Not suited for replacing existing targets
    "target_engineering": {
        "iter0_target_engineer": (
            "CLEAN_TARGET_DATABASE",
            """
            SELECT * FROM targets
            """,
            'target_engineering',
            'targets'
            )
        },
    
    "scaler_save_dir": (r'C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning'
                         r'\projects\iter0\training\scalers'),
    
    # Specify the db_path to scale features for
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) List of table names to scale
    "feature_scaling": {
        "iter0_features" : (
            "CLEAN_FEATURE_DATABASE",
            ['hourly_features']
            )
        },
    
    "prepped_data_dir": (r'C:\Users\brand\OneDrive\Blase Capital Mgmt'
                         r'\deep_learning\projects\iter0\training\prepped_data'),
    
    # Specify the db_path to scale features for
    # Keys: meta
    #   Items:
    #       1) .env database path reference
    #       2) Dict of table names to specify if they need to be scaled or reshaped
    #       3) Path to save scale files
    "prep_and_save": {
        "iter0_features": (
            "CLEAN_FEATURE_DATABASE",
            {
                'hourly_features': {
                    # observed ohlc may be skewed relative to one another in a 
                    # window (ie low originally being higher than close but 
                    # then altered after scaling)
                    'scaler': True, 
                    'keep_primary_key': False
                    },
                'dense_features': {
                    'scaler': False,
                    'keep_primary_key': False
                    }
                }
            ),
        "iter0_targets": (
            "CLEAN_TARGET_DATABASE",
            {
                'targets': {
                    'scaler': False,
                    'weights_dict': 'target',
                    'keep_primary_key': True
                    }
                }
            )
        },
    "weight_dict_save_path": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                              r"\deep_learning\projects\iter0\training"
                              r"\weights_dict\target_weights_dict.json"), 
            
    # Args for training
    # Reshape (number of windows (48 hours of ohlc=48), size of windows (ohlc=4))
    "feature_categories": 
        {
            'hourly': {
                'reshape': True,
                'shape': (48, 4)
                },
            'dense': {
                'reshape': False,
                'shape': None
                }
        },
    "model_modules_path": (r'C:\Users\brand\OneDrive'
                           r'\Blase Capital Mgmt\deep_learning'
                           r'\projects\iter0\training'
                           r'\iter0_training.py'),
    "model_function": 'create_model',
    "callback_function": 'AggregateCallbacks',
    "model_args": {
        'n_hours_back_hourly': 192 // 4, # len(features) // len(window)
        'n_ohlc_features': 4, # window
        'l2_strength': 0.01, 
        'dropout_rate': 0.5,
        'n_dense_features': 47, 
        'activation': 'tanh', 
        'n_targets': 3, 
        'output_activation': 'softmax',
        'initial_bias': True
        },
    
    # Use for both initial_bias and class_weights
    "weight_dict_path": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                         r"\deep_learning\projects\iter0\training"
                         r"\weights_dict\target_weights_dict - Copy.json"), 
    "data_dir": (r'C:\Users\brand\OneDrive\Blase Capital Mgmt'
                 r'\deep_learning\projects\iter0\training\prepped_data'),    
    # "custom_loss": {"custom_loss_path": (r'C:\Users\brand\OneDrive'
    #                                      r'\Blase Capital Mgmt\deep_learning'
    #                                      r'\projects\iter0\training'
    #                                      r'\iter0_training.py'),
    #                 "module_name": "custom_loss"},
    "custom_loss": {},
    "use_weight_dict": True,
    # The layer is the key and the loss is the item
    "loss": {"output_layer": "sparse_categorical_crossentropy"},
    "metrics": ["accuracy"],
    "optimizer": {"type": "adam", "learning_rate": 0.001, "clipvalue": 1.0},
    "epochs": 100,
    "batch_size": 4096,
    "iteration_dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                      r"\deep_learning\projects\iter0\training\iterations"),
    # Specify the file paths of the iteration config.py and model architecture file
    "requirements_paths": {
        "config": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
                   r"\projects\iter0\training\config.py"),
        "model": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
                  r"\projects\iter0\training\iter0_training.py")
        }
    }
        