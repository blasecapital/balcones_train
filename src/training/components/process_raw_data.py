# process_raw_data.py


import pandas as pd
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os
from typing import Union

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from config import config
from utils import public_method


class ProcessRawData:
    def __init__(self, df_dict: dict):
        # The initialized object
        self.df_dict = df_dict
        
        # Config specs
        self.module_path = config.get("data_processing_modules_path")
        self.filter_function = config.get("filter_function")
        self.primary_key = config.get("primary_key")
        self.feature_eng_list = config.get("feature_engineering")
        self.target_eng_list = config.get("target_engineering")
        self.features = config.get("define_features")
        self.targets = config.get("define_targets")
        self.category_index = config.get("category_index")
        self.scaler_save_path = config.get("scaler_save_path")
        self.reshape = config.get("reshape")
        
    def _import_function(self, function_name):
        """
        Dynamically import a module specified in `self.module_path` and 
        return the function from the arg.
        """
        spec = importlib.util.spec_from_file_location(
            "module", self.module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
        # Access the specified filter function dynamically
        filter_function = getattr(module, function_name)
        
        return filter_function
    
    @public_method
    def filter_indices(self):
        """
        Accept custom filter module and remove indices across all DataFrames
        in the df_dict. Returns the updated df_dict with filtered DataFrames.
        """
        # Import the filter function
        filter_function = self._import_function(self.filter_function)
        
        # Initialize an empty set to accumulate indices
        nan_indices_union = set()
        
        # Collect NaN indices from each DataFrame
        for df_key, df in self.df_dict.items():
            # Get indices from the custom filter function
            filter_nan_indices = filter_function(df)
            
            # Validate the output of the filter function
            if not isinstance(filter_nan_indices, (list, set, pd.Index)):
                raise ValueError(f"Filter function must return a list, set, or "
                                 f"Index. Got {type(filter_nan_indices)}")
            
            # Identify rows entirely filled with NaN and add their indices
            all_nan_indices = df[df.isna().all(axis=1)].index
            
            # Combine indices from the filter function and entirely NaN rows
            nan_indices = set(filter_nan_indices).union(set(all_nan_indices))
            
            # Update the union set with the new indices
            nan_indices_union.update(nan_indices)
    
        # Apply the union of NaN indices to all DataFrames in df_dict
        for df_key in self.df_dict.keys():
            self.df_dict[df_key].drop(
                index=nan_indices_union, inplace=True, errors="ignore")
            print(f"Filtered {df_key}:")
            print(self.df_dict[df_key])
            
        # Return the updated DataFrame dictionary
        return self.df_dict
        
    def _get_dataframes(self, dfs: Union[str, list] = "all"):
            """
            Helper function to retrieve the DataFrames to process.
            """
            if dfs == "all":
                return self.df_dict.items()
            elif isinstance(dfs, str):
                return [(dfs, self.df_dict[dfs])]
            elif isinstance(dfs, list):
                return [(key, self.df_dict[key]) for key in dfs]
            else:
                raise ValueError("Invalid dfs argument. Must be 'all', a string,"
                                 " or a list.")

    def _get_df_columns(self, df, columns: Union[str, list, str] = "all"):
        """
        Helper function to retrieve the columns to process.
        """
        if columns == "all":
            return df.columns
        elif isinstance(columns, str):
            return [columns]
        elif isinstance(columns, list):
            return [col for col in columns if col in df.columns]
        else:
            raise ValueError("Invalid columns argument. Must be 'all', a string,"
                             " or a list.")

    @public_method
    def print_cols(self, dfs="all", columns="all"):
        """
        Print the specified columns for the selected DataFrames.

        Args:
            dfs (str | list): DataFrames to process ('all', a single name, or a list of names).
            columns (str | list): Columns to print ('all', a single name, or a list of names).
        """
        target_dfs = self._get_dataframes(dfs)

        for name, df in target_dfs:
            print(f"\nDataFrame: {name} columns:")
            target_columns = self._get_df_columns(df, columns)
            print(df[target_columns].columns)

    @public_method
    def plot_col(self, dfs="all", columns="all"):
        """
        Plot the specified columns for the selected DataFrames.

        Args:
            dfs (str | list): DataFrames to process ('all', a single name, or a list of names).
            columns (str | list): Columns to plot ('all', a single name, or a list of names).
        """
        plot_dfs = self._get_dataframes(dfs)

        for name, df in plot_dfs:
            df_columns = [col for col in self._get_df_columns(df, columns) 
                              if col not in self.primary_key]
            
            # Replace inf and -inf with NaN before plotting
            df = df.replace([np.inf, -np.inf], np.nan)

            for col in df_columns:
                plt.figure(figsize=(8, 6))
                if df[col].dtype == "object" or len(df[col].unique()) < 10:
                    # Categorical data: Histogram
                    plt.hist(df[col].dropna(), bins=10, edgecolor="black")
                    plt.title(f"Histogram for {col} in {name}")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                else:
                    # Numerical data: Violin-like plot (using Matplotlib's boxplot for simplicity)
                    plt.boxplot(df[col].dropna(), vert=False)
                    plt.title(f"Box Plot for {col} in {name}")
                    plt.xlabel("Value")
    
                plt.tight_layout()
                plt.show()

    @public_method
    def describe_cols(self, dfs="all", columns="all"):
        """
        Describe the specified columns for the selected DataFrames. Exluding
        the columns specified in the config file's primary_key list.

        Args:
            dfs (str | list): DataFrames to process ('all', a single name, or a list of names).
            columns (str | list): Columns to describe ('all', a single name, or a list of names).
        """
        describe_dfs = self._get_dataframes(dfs)

        for name, df in describe_dfs:
            print(f"\nDataFrame: {name} - Descriptive Statistics")
            df_columns = [col for col in self._get_df_columns(df, columns) 
                              if col not in self.primary_key]
            print(df[df_columns].describe(include="all"))
        
    @public_method
    def engineering(self, mode='all'):
        """
        Apply engineering functions (feature or target) to their corresponding DataFrames.
    
        Args:
            mode (str): 'all' (default) to perform both feature and target engineering,
                        or 'feature'/'target' to perform only the respective type.
    
        Returns:
            dict: Updated dictionary of DataFrames.
        """
        if mode not in ["feature", "target", "all"]:
            raise ValueError("Invalid mode. Must be 'feature', 'target', or 'all'.")
    
        # Determine which lists to process based on the mode
        eng_lists = []
        if mode in ["feature", "all"]:
            eng_lists.append((self.feature_eng_list, "feature"))
        if mode in ["target", "all"]:
            eng_lists.append((self.target_eng_list, "target"))
    
        # Iterate over the selected engineering lists
        for eng_list, eng_type in eng_lists:
            for eng_function, df_key in eng_list:
                # Import the engineering function dynamically
                eng_func = self._import_function(eng_function)
    
                # Retrieve the DataFrame
                if df_key not in self.df_dict:
                    raise KeyError(f"DataFrame key '{df_key}' not found in df_dict.")
    
                # Apply the engineering function
                eng_df = self.df_dict[df_key]
                self.df_dict[df_key] = eng_func(eng_df)
    
                print(f"{eng_type.capitalize()} engineering applied to DataFrame: {df_key}")
        
        return self.df_dict
    
    @public_method
    def encode_categories(self):
        """
        If building classification models, use this to encode the target column
        if needed.
        """      
        dfs_to_encode = set()
        for cat_map, df_key in self.category_index:
            dfs_to_encode.add(df_key)
            
        for df_key in dfs_to_encode:
            # Get the target column associated with the df_key
            target = None
            for target_name, target_df_key in self.targets:
                if target_df_key == df_key:
                    target = target_name
                    break
            
            # Check if target was found for the DataFrame
            if not target:
                raise ValueError(f"No target column found for DataFrame key '{df_key}' in targets.")
            
            # Perform encoding using the category mapping
            for cat_map, cat_df_key in self.category_index:
                if cat_df_key == df_key:
                    df = self.df_dict[df_key]
                    if target not in df.columns:
                        raise ValueError(f"Target column '{target}' not found in DataFrame '{df_key}'.")
                    
                    # Encode the target column using the category mapping
                    df[target] = df[target].map(cat_map)
                    print(f"Encoded target column '{target}' in DataFrame '{df_key}'.")
                    break
                
    def _define_features(self):
        """
        Apply each DataFrame's feature list creation function and return a 
        list of sets containing the feature list and DataFrames name.
        """
        feature_dict = {}
        for feat_function, df_key in self.features:
            # Import the feature listing function dynamically
            feat_func = self._import_function(feat_function)
            
            # Retrieve the DataFrame
            if df_key not in self.df_dict:
                raise KeyError(f"DataFrame key '{df_key}' not found in df_dict.")
                
            feat_list = feat_func(self.df_dict[df_key])
            
            feature_dict[df_key] = feat_list
        
        print("Feature dict created:", feature_dict)
        
        return feature_dict
    
    def _feature_typing(self):
        """ 
        Convert each DataFrame's feature column set into a numpy array.
        """
        feature_dict = self._define_features()
        
        feature_array_dict = {}
        
        for df_key, features in feature_dict.items():
            # Retrieve the DataFrame using the key
            if df_key not in self.df_dict:
                raise KeyError(f"DataFrame '{df_key}' not found in df_dict.")
            
            df = self.df_dict[df_key]
            
            # Ensure all features exist in the DataFrame
            missing_features = [feature for feature in features if feature not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features {missing_features} in DataFrame '{df_key}'.")
            
            # Convert the feature columns to a NumPy array
            arr = df[features].values.astype(np.float32)
            feature_array_dict[df_key] = arr
        
        print("Completed feature type conversion:", feature_array_dict)
        
        return feature_array_dict
        
    def _split(self, train=0.6, val=0.2, test=0.2):
        """ 
        Calculate the train, validation, and test indices for splitting datasets.
        
        Args:
            train (float): Proportion of data for training.
            val (float): Proportion of data for validation.
            test (float): Proportion of data for testing.
    
        Returns:
            dict: A dictionary containing train, val, and test indices.
        """
        # Validate split ratios
        if not np.isclose(train + val + test, 1.0):
            raise ValueError("The sum of train, val, and test splits must equal 1.0.")
    
        # Helper to ensure all lengths are equal
        lengths = set()
        for df_key, df in self.df_dict.items():
            df_len = len(df)
            lengths.add(df_len)
    
        if len(lengths) != 1:
            raise ValueError("Not all dataframes are of equal lengths.")
    
        n_samples = lengths.pop()  # Extract the single unique length
    
        # Calculate split indices
        n_train = int(train * n_samples)
        n_val = int(val * n_samples)
    
        # Create split index dictionary
        splits = {
            "train": n_train,
            "val": n_val,
            "test": n_samples - n_train - n_val,
        }
    
        print(f"Data split indices calculated: {splits}")
        return splits
    
    def _save_scaler(self, scaler, df_key):
        """
        Save the fitted scaler to a file.
        
        Args:
            scaler (object): Fitted scaler.
            df_key (str): DataFrame key.
        """
        # Ensure the save path exists
        if not os.path.exists(self.scaler_save_path):
            os.makedirs(self.scaler_save_path)
    
        # Save the scaler
        file_name = f'scaler_{df_key}.pkl'
        save_path = os.path.join(self.scaler_save_path, file_name)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to: {save_path}")
        
    def _reshape(self, feature_arr, reshape_config):
        """
        Reshape a feature array based on configuration.
    
        Args:
            feature_arr (np.ndarray): Feature array to reshape.
            reshape_config (tuple): Reshape configuration (samples, timesteps, features, df_key).
    
        Returns:
            np.ndarray: Reshaped feature array.
        """
        samples, timesteps, features, df_key = reshape_config
    
        # Automatic calculation for `samples` if set to -1
        if samples == -1:
            samples = feature_arr.shape[0] // timesteps
    
        # Validate the reshape dimensions
        expected_size = samples * timesteps * features
        if feature_arr.size != expected_size:
            raise ValueError(
                f"Feature array for '{df_key}' cannot be reshaped into "
                f"(samples={samples}, timesteps={timesteps}, features={features}). "
                f"Expected size: {expected_size}, got: {feature_arr.size}."
            )
    
        # Perform reshaping
        reshaped = feature_arr.reshape(samples, timesteps, features)
        print(f"Reshaped data for '{df_key}' to shape {reshaped.shape}.")
    
        return reshaped
        
    @public_method
    def prepare_features(self, mode='robust'):
        """ 
        For each dataframe, split using the _split method, fit scalers on
        the training data, save the scaler, and transform the data.
        
        Args:
            mode (str): Either 'robust' or 'standard'
        
        Returns:
            seg_data_dict (dict): Dictionary of dictionaries, each dataframe
                                  contains train, val, and test data.
        """
        # Validate mode
        if mode not in ['robust', 'standard']:
            raise ValueError("Mode must be 'robust' or 'standard'.")
    
        # Select scaler
        scaler_class = RobustScaler if mode == 'robust' else StandardScaler
        
        seg_data_dict = {}
        
        splits = self._split()
        
        # Convert split counts into cumulative indices
        train_end = splits['train']
        val_end = train_end + splits['val']
        test_end = val_end + splits['test']
    
        # Iterate over feature arrays
        for df_key, feature_arr in self._feature_typing().items():
            # Initialize the scaler
            scaler = scaler_class()
    
            # Ensure the feature array has enough samples
            if feature_arr.shape[0] < test_end:
                raise ValueError(f"Feature array for '{df_key}' is too small for "
                                 "the specified splits.")
    
            # Segment the data using calculated indices
            train_data = feature_arr[:train_end]
            val_data = feature_arr[train_end:val_end]
            test_data = feature_arr[val_end:test_end]
    
            # Fit the scaler on training data
            scaler.fit(train_data)
            
            # Ensure the save path exists
            if not os.path.exists(self.scaler_save_path):
                os.makedirs(self.scaler_save_path)
            
            # Save the scaler
            self._save_scaler(scaler, df_key)
    
            # Scale each segment
            scaled_train = scaler.transform(train_data)
            scaled_val = scaler.transform(val_data)
            scaled_test = scaler.transform(test_data)
            
            # Check if reshaping is needed for the current DataFrame
            reshape_config = next((item for item in self.reshape if item[3] == df_key), None)
            if reshape_config:
                scaled_train = self._reshape(scaled_train, reshape_config)
                scaled_val = self._reshape(scaled_val, reshape_config)
                scaled_test = self._reshape(scaled_test, reshape_config)
    
            # Store scaled segments in the dictionary
            seg_data_dict[df_key] = {
                'train': scaled_train,
                'val': scaled_val,
                'test': scaled_test
            }
            
            print(f"Scaling completed for DataFrame '{df_key}'.")
        
        self.seg_data_dict = seg_data_dict
        return seg_data_dict
    
    def _handle_target_encoding(self, df, target, df_key):
        """
        Encodes categorical targets in the DataFrame if necessary.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the target column.
            target (str): The target column to encode.
            df_key (str): The key for the DataFrame in df_dict.
        """
        try:
            print(f"Encoding categories for target '{target}' in DataFrame '{df_key}'.")
            self.encode_categories()
        except Exception as e:
            raise ValueError(
                f"Failed to encode categories for target '{target}' in DataFrame '{df_key}': {e}"
            )
    
    def _target_typing(self):
        """ 
        Convert each DataFrame's target column set into a numpy array.
        """
        target_array_dict = {}
        
        for target_set in self.targets:
            target = target_set[0]
            df_key = target_set[1]
            # Retrieve the DataFrame using the key
            if df_key not in self.df_dict:
                raise KeyError(f"DataFrame '{df_key}' not found in df_dict.")
            
            df = self.df_dict[df_key]
            
            # Ensure target exists in the DataFrame
            if target not in df.columns:
                raise ValueError(f"Missing target {target} in DataFrame '{df_key}'.")
            
            # Convert the target columns to a NumPy array
            try:
                arr = df[target].values.astype(np.float32)
            except (ValueError, TypeError) as e:
                print(f"Conversion failed for target '{target}' in DataFrame '{df_key}': {e}")
                # Attempt to encode categories if conversion fails
                self._handle_target_encoding(df, target, df_key)
                # Retry the conversion after encoding
                try:
                    arr = df[target].values.astype(np.float32)
                except (ValueError, TypeError) as final_error:
                    raise ValueError(
                        f"Failed to convert target '{target}' in DataFrame '{df_key}' "
                        f"even after category encoding: {final_error}"
                    )
            target_array_dict[df_key] = arr
        
        print("Completed target type conversion.")
        
        return target_array_dict
    
    @public_method
    def prepare_targets(self):
        """
        For each dataframe, split using the _split method.
        
        Returns:
            split_target_dict (dict): Dictionary of dictionaries, each dataframe
                                      contains train, val, and test target data.
        """
        split_target_dict = {}
        
        target_array_dict = self._target_typing()
        
        splits = self._split()
        
        # Convert split counts into cumulative indices
        train_end = splits['train']
        val_end = train_end + splits['val']
        test_end = val_end + splits['test']
        
        for df_key, target_arr in target_array_dict.items():
            # Segment the data using calculated indices
            train_data = target_arr[:train_end]
            val_data = target_arr[train_end:val_end]
            test_data = target_arr[val_end:test_end]
            
            # Store scaled segments in the dictionary
            split_target_dict[df_key] = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
        
        print("Completed target splitting.")
        
        self.split_target_dict = split_target_dict
        return split_target_dict            
        
    @public_method
    def process_raw_data(
            self,
            filter_indices=False,
            engineering=False,
            engineering_mode='feature',
            encode_categories=False,
            prepare_features=True,
            prepare_targets=True
            ):
        """
        Orchestrate the entire training data preprocessing using all 
        essential methods by default.
        """
        # Save configurations as attributes
        self.filter_indices_flag = filter_indices
        self.engineering_flag = engineering
        self.engineering_mode = engineering_mode
        self.encode_categories_flag = encode_categories
        self.prepare_features_flag = prepare_features
        self.prepare_targets_flag = prepare_targets
        
        if filter_indices:
            self.filter_indices()
        if engineering:
            self.engineering(engineering_mode)
        if encode_categories:
            self.encode_categories()
        if prepare_features:
            self.prepare_features()
        if prepare_targets:
            self.prepare_targets()
            
        training_data = {
            "features": self.seg_data_dict,
            "targets": self.split_target_dict,
            "process_data_config": {
                "filter_indices": self.filter_indices_flag,
                "engineering": self.engineering_flag,
                "engineering_mode": self.engineering_mode,
                "encode_categories": self.encode_categories_flag,
                "prepare_features": self.prepare_features_flag,
                "prepare_targets": self.prepare_targets_flag,
            },
        }
        
        return training_data
        