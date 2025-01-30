# prep_data.py


import pandas as pd
import numpy as np
import importlib.util
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os
from typing import Union
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from .load_training_data import LoadTrainingData
from utils import EnvLoader, public_method


class PrepData:
    def __init__(self):
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        # Config specs
        self.source_query = self.config.get('source_query')
        self.project_dir = self.config.get('project_directory')
        
        self.module_path = self.config.get("data_processing_modules_path")
        self.filter_function = self.config.get("filter_function")
        self.primary_key = self.config.get("primary_key")
        self.feature_eng_list = self.config.get("feature_engineering")
        self.target_eng_list = self.config.get("target_engineering")
        self.features = self.config.get("define_features")
        self.targets = self.config.get("define_targets")
        self.category_index = self.config.get("category_index")
        self.scaler_save_path = self.config.get("scaler_save_path")
        self.reshape = self.config.get("reshape")
        
        # Initialize data loader
        self.ltd = LoadTrainingData()
        
    @public_method
    def prep(self):
        """
        Facilitate the data preparation process:
            1) Engineer, encode, and store features.
            2) Shape the data.
            3) Split the data.
            4) Normalize the data.
            5) Type the data.
            6) Store the data.
        
        The end product is a directory with train, val, and test subdirectories
        filled with batched files (.npy, .h5, .parquet, or TFRecord) which 
        enables out-of-the-box data for training and model validation.
        """
        print("This is the main public function!")