# config.py


config = {
    # Specify the backtest evaluation directory
    "backtest": {
        "dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
                r"\projects\iter0\training\iterations\20_02_2025_15_36_59"),
        "model_weights": "epoch0_trainLoss_1.2077_trainAcc_0.6159_valLoss_1.1058_valAcc_0.6746.h5",
        "create_model_module_src": "iter0_training.py",
        "model_module": "create_model",
        "model_config_src": "config.py",
        "config_dict_name": "config",
        "model_args": "model_args",
        "feature_categories": "feature_categories",
        "save_pred_dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                          r"\deep_learning\projects\iter0\evaluation\predictions"),
        "primary_keys": ("date", "pair")
        },
    "explain": {
        "dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
                r"\projects\iter0\training\iterations\20_02_2025_15_36_59"),
        "model_weights": "epoch0_trainLoss_1.2077_trainAcc_0.6159_valLoss_1.1058_valAcc_0.6746.h5",
        "create_model_module_src": "iter0_training.py",
        "model_module": "create_model",
        "model_config_src": "config.py",
        "config_dict_name": "config",
        "model_args": "model_args",
        "file_num": 44, # Specify which group in prepped_data to load
        "class_names": [0,1,2],
        "contains_categorical_features": True,
        "categorical_feature_id": "pair_",
        "sample_num": 38, # Specify which entry index in the file to explain,
        "id_cols": ["date", "pair"],
        "prediction_dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                           r"\deep_learning\projects\iter0\evaluation\predictions"
                           r"\20_02_2025_15_36_59_epoch0.db")
        },
    "metrics": {
        "db": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
               r"\projects\iter0\evaluation\predictions\19_02_2025_17_31_29_epoch3.db"),
        "query": """
        SELECT * FROM predictions
        WHERE pair IN ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD')
        """,
        "metric_categories": {
            "split_col": "split",
            "asset_col": 'pair',
            "target_col": 'predicted_category',
            },
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "metrics": ['accuracy', 'precision', 'recall', 'f1_score', 'log_loss', 'roc_auc'],
        },
    "calibration": {
        "db": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
               r"\projects\iter0\evaluation\predictions\19_02_2025_17_31_29_epoch3.db"),
        "query": """
        SELECT * FROM predictions
        WHERE pair IN ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD')
        """,
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "cal_categories": {
            "split_col": "split",
            "asset_col": 'pair',
            "target_col": 'predicted_category',
            }
        },
    "candidates": {
        "db": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
               r"\projects\iter0\evaluation\predictions\20_02_2025_15_36_59_epoch0.db"),
        # ***GENERALLY ENSURE THE DATASET IS FILTERED BY VAL AND TEST SPLITS***
        "query": """
        SELECT * FROM predictions
        WHERE split IN ('val', 'test') 
        AND pair IN ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD')
        """,
        "y_true": "target",
        "y_pred": "predicted_category",
        "y_conf": "confidence",
        "class_filter": {
            "column_name": "predicted_category",
            "classes": [1,2]
            },
        "asset_col": "pair",
        # The custom function must use df as an arguement
        "custom_func": True,
        "custom_func_path": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt"
                             r"\deep_learning\projects\iter0\evaluation"
                             r"\iter0_eval.py"),
        "custom_func_name": "calculate_running_profit"
        }
    }