# load_eval_data.py


import sqlite3
import pandas as pd
import importlib.util
import os

from utils import EnvLoader, public_method
from config import config


class LoadEvalData:
    def __init__(self, model_name):
        """
        Initialize the LoadEvalData class with configuration and utility modules.
        """
        self.model_name = model_name
        self.eval_dict = config
        self.original_config = self._load_original_config()
        self.env_loader = EnvLoader()
        
        self.source_query = self.original_config.get('source_query')
        self.primary_key = self.original_config.get('primary_key')
    
    def _load_original_config(self):
        """
        Dynamically load the original `config` module from the requirements directory.

        Returns:
            dict: The `config` dictionary from the original file.
        
        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ImportError: If the module cannot be loaded.
        """
        original_config_path = self.eval_dict[self.model_name]["requirements_directory"]
        
        if not os.path.exists(original_config_path):
            raise FileNotFoundError(f"The configuration file does not exist at: {original_config_path}")
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("config", original_config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Return the config dictionary
        return config_module.config
        
    def _load_from_database(self, db_path, query):
        """
        Internal helper function to load data from a database.
        """
        if not query:
            raise ValueError("Query is empty or not provided.")
        
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql(query, conn)
                print("Data loaded successfully...", df.head())
                return df
        except Exception as e:
            raise RuntimeError(f"Error loading data from the database: {e}")
            
    def _align(self, df_dict):
        """
        Ensure all DataFrames are aligned on the primary key and have the same length.
    
        Args:
            df_dict (dict): A dictionary where keys are descriptive names and values are DataFrames.
    
        Returns:
            dict: A dictionary of aligned DataFrames with identical indices based on the primary key.
        """
        if not self.primary_key:
            raise ValueError("Primary key is not defined in the configuration.")
    
        # Check for missing primary key columns in each DataFrame
        for key, df in df_dict.items():
            missing_keys = [col for col in self.primary_key if col not in df.columns]
            if missing_keys:
                raise ValueError(f"DataFrame '{key}' is missing primary key columns: {missing_keys}")
    
        # Determine the union of all primary keys across DataFrames
        all_keys = pd.concat([df[self.primary_key] for df in df_dict.values()]).drop_duplicates()
        all_keys.set_index(self.primary_key, inplace=True)
    
        # Reindex each DataFrame to align with the union of primary keys
        aligned_dict = {}
        for key, df in df_dict.items():
            df.set_index(self.primary_key, inplace=True, drop=False)
            aligned_df = all_keys.join(df, how="left").reset_index(drop=True)
            aligned_dict[key] = aligned_df
    
            # Ensure the lengths match
            if len(aligned_df) != len(all_keys):
                raise ValueError(f"DataFrame '{key}' could not be fully aligned with the primary key index.")
    
        print("All DataFrames are aligned on the primary key and have the same length.")
        return aligned_dict
    
    @public_method
    def load_data(self):
        """
        Load each dataframe based on the config's source_query sets.
    
        Returns:
            dict: A dictionary where keys are descriptive names for the DataFrames
                  and values are the loaded DataFrames.
        """
        print("Beginning to load data...")
    
        df_dict = {}
    
        # Iterate through the source_query list from config
        for i, (source_type, query, meta) in enumerate(self.source_query):
            # Retrieve database path from environment variables or config
            db_path = self.env_loader.get(source_type)
    
            # Load data from the database
            try:
                df = self._load_from_database(db_path, query)
            except Exception as e:
                raise RuntimeError(f"Failed to load data for source {source_type}: {e}")
    
            # Use descriptive keys for the dictionary
            df_key = f"df{i}" if not meta else f"{meta}"
            df_dict[df_key] = df
        
        # Ensure each dataframe is aligned
        if len(df_dict) > 1:
            df_dict = self._align(df_dict)
            
        print(f"Successfully loaded {len(df_dict)} dataframes.")
        return df_dict
        