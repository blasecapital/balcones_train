# config.py


config = {
    # Specify the backtest evaluation directory
    "backtest": {
        "dir": (r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning"
                r"\projects\iter0\training\iterations\13_02_2025_09_58_46"),
        "model_weights": "epoch14_trainLoss_0.9961_trainAcc_0.5603_valLoss_0.8850_valAcc_0.6180.h5",
        "create_model_module_src": "iter0_training.py",
        "model_module": "create_model",
        "model_config_src": "config.py",
        "config_dict_name": "config",
        "model_args": "model_args",
        "feature_categories": "feature_categories"
        },
    }