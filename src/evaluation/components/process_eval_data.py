# process_eval_data.py


import pandas as pd
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import EnvLoader, public_method


class ProcessEvalData:
    def __init__(self):
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("EVALUATE_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
