# process_raw_data.py


import pandas as pd
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os
from typing import Union
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import mplfinance as mpf

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from .load_training_data import LoadTrainingData
from utils import EnvLoader, public_method


class ProcessRawData:
    def __init__(self):#, df_dict: dict):
        # The initialized object
        #self.df_dict = df_dict
        
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
        
    ### FUNCTIONS WHICH WILL STILL WORK WITH CHUNKING REVISION
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
    
    ### NEW PROCESS
    def _get_progress_log(self, key):
        """
        Retrieve the progress log for the specified key. If the log does not exist, initialize it.
    
        Args:
            key (str): The key identifying the dataset being processed.
    
        Returns:
            dict: The progress log for the specified key.
        """
        if not hasattr(self, "_progress_logs"):
            self._progress_logs = {}  # Initialize a dictionary to store progress logs
    
        if key not in self._progress_logs:
            # Initialize the progress log for the given key
            self._progress_logs[key] = {
                "running_sum": {},
                "running_square_sum": {},
                "running_count": {},
                "running_min": {},
                "running_max": {},
                "bin_edges": {},
                "bin_frequencies": {},
                "chunk_stats": {},
                "bad_keys": [],
                "final_stats": None,
            }
        
        return self._progress_logs[key]
    
    def _update_bin_frequencies(self, feature, data, bin_edges, bin_frequencies):
        """
        Update bin frequencies for a feature based on the current chunk of data.
    
        Args:
            feature (str): The feature/column name.
            data (DataFrame): The current chunk of data.
            bin_edges (array): The edges of the bins.
            bin_frequencies (dict): A dictionary storing the frequencies of values in each bin,
                                    including underflow and overflow bins.
    
        Example:
            bin_edges = [0, 10, 20, 30]
            bin_frequencies = {"underflow": 0, "overflow": 0, 0: 0, 1: 0, 2: 0}
        """
        if feature not in data.columns:
            print(f"Feature '{feature}' not found in data columns.")
            return
    
        # Extract values for the feature, dropping NaNs
        values = data[feature].dropna().values
    
        # Digitize the values into bins
        bin_indices = np.digitize(values, bin_edges, right=False)
    
        # Ensure bins are initialized
        bin_frequencies.setdefault(feature, {"underflow": 0, "overflow": 0})
        for i in range(len(bin_edges) - 1):
            bin_frequencies[feature].setdefault(i, 0)
    
        # Update bin frequencies
        for value, idx in zip(values, bin_indices):
            if idx == 0:
                # Value is below the smallest bin
                bin_frequencies[feature]["underflow"] += 1
            elif idx > len(bin_edges) - 1:
                # Value is above the largest bin
                bin_frequencies[feature]["overflow"] += 1
            else:
                # Value falls within a valid bin
                bin_frequencies[feature][idx - 1] += 1
    
    def _calculate_percentiles(self, feature, bin_edges, bin_frequencies, total_count):
        """
        Calculate percentiles (e.g., 25%, 50%, 75%) using binned data.
    
        Args:
            feature (str): The feature/column name.
            bin_edges (array): The edges of the bins.
            bin_frequencies (dict): A dictionary storing bin frequencies, including underflow and overflow bins.
            total_count (int): The total number of valid values for the feature.
    
        Returns:
            dict: A dictionary containing percentiles (25%, 50%, 75%).
        """
        if total_count == 0:
            return {"25%": None, "50%": None, "75%": None}
    
        # Extract bin frequencies as a list in bin order
        frequencies = [bin_frequencies.get(i, 0) for i in range(len(bin_edges) - 1)]
        cumulative_frequency = np.cumsum(frequencies)  # Cumulative frequency for interpolation
        
        percentiles = {}
        for percentile, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
            target_count = total_count * percentile
    
            # Find the bin where the target count falls
            bin_idx = np.searchsorted(cumulative_frequency, target_count)
    
            if bin_idx == 0:
                # If the target count is in the first bin
                percentiles[label] = bin_edges[0]
            elif bin_idx >= len(bin_edges) - 1:
                # If the target count is in the last bin
                percentiles[label] = bin_edges[-1]
            else:
                # Interpolate within the bin
                bin_start = bin_edges[bin_idx - 1]
                bin_end = bin_edges[bin_idx]
                bin_frequency = frequencies[bin_idx - 1]
                prev_cumulative = cumulative_frequency[bin_idx - 1]
                curr_cumulative = cumulative_frequency[bin_idx]
    
                if bin_frequency > 0:
                    # Linear interpolation within the bin
                    interpolated_value = bin_start + (
                        (target_count - prev_cumulative) / (curr_cumulative - prev_cumulative)
                    ) * (bin_end - bin_start)
                    percentiles[label] = interpolated_value
                else:
                    # If the bin has zero frequency, fallback to the bin start
                    percentiles[label] = bin_start
    
        return percentiles
    
    def _finalize_describe_report(self, key, progress_log):
        """
        Finalize and generate aggregated statistics after processing all chunks.
        """
        final_stats = {}
    
        for feature in progress_log["running_sum"]:
            count = progress_log["running_count"][feature]
            if count == 0:
                continue
    
            # Calculate mean, std, min, and max
            mean = progress_log["running_sum"][feature] / count
            variance = (
                progress_log["running_square_sum"][feature] / count - mean ** 2
            )
            std_dev = np.sqrt(variance)
            min_value = progress_log["running_min"][feature]
            max_value = progress_log["running_max"][feature]
    
            # Calculate percentiles using bin data
            percentiles = self._calculate_percentiles(
                feature,
                progress_log["bin_edges"][feature],
                progress_log["bin_frequencies"][feature],
                count,
            )
    
            # Store aggregated statistics
            final_stats[feature] = {
                "count": count,
                "mean": mean,
                "std": std_dev,
                "min": min_value,
                "25%": percentiles.get("25%"),
                "50%": percentiles.get("50%"),
                "75%": percentiles.get("75%"),
                "max": max_value,
            }
    
        # Save the final stats to the progress log
        progress_log["final_stats"] = final_stats
        
    def _initialize_bins_from_sql(self, feature_list, key):
        """
        Initialize bin edges for each feature by querying the global MIN and MAX using SQL, 
        incorporating any filtering conditions from the source query.
        
        Args:
            feature_list (list): List of feature/column names.
            key (str): The key identifying the dataset, used to derive the table name and database path.
        
        Returns:
            dict: A dictionary containing bin edges for each feature.
        """
        # Get the database path from the environment loader
        db_path = self.env_loader.get(self.source_query[key][0])
        query = self.source_query[key][1]
        
        # Extract table name and WHERE clause
        table_name = query.split("FROM")[1].split("WHERE")[0].strip()
        where_clause = query.split("WHERE")[1].strip() if "WHERE" in query else ""
    
        def process_feature(feature):
            """
            Helper function to query MIN and MAX for a single feature.
            """
            try:
                # Connect to the database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
    
                # Build the query for MIN and MAX, including the WHERE clause if it exists
                min_max_query = f"SELECT MIN({feature}) AS min_val, MAX({feature}) AS max_val FROM {table_name}"
                if where_clause:
                    min_max_query += f" WHERE {where_clause}"
                
                # Execute the query and fetch results
                cursor.execute(min_max_query)
                result = cursor.fetchone()
                conn.close()
    
                if result and result[0] is not None and result[1] is not None:
                    min_val = float(result[0])
                    max_val = float(result[1])
                    return feature, np.linspace(min_val, max_val, num=101)
                else:
                    raise ValueError(f"No valid data found for feature '{feature}' in table '{table_name}' with filter '{where_clause}'.")
            except Exception as e:
                print(f"Error initializing bins for feature '{feature}': {e}")
                # Default fallback if MIN and MAX cannot be computed
                return feature, np.linspace(0, 1, num=101)
    
        # Use ThreadPoolExecutor to parallelize the feature processing with progress bar
        bin_edges = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_feature, feature): feature for feature in feature_list}
            with tqdm(total=len(futures), desc="Processing Features", unit="feature") as pbar:
                for future in as_completed(futures):
                    feature, edges = future.result()
                    bin_edges[feature] = edges
                    pbar.update(1)
    
        return bin_edges
    
    def _display_descriptive_stats(self, progress_log):
        """
        Display final stats for each feature as a well-structured DataFrame.
        """
        final_stats = progress_log['final_stats']
        
        # Create a DataFrame where rows are features and columns are statistics
        stats_df = pd.DataFrame(final_stats).T  # Transpose so features are rows
        
        # Temporarily adjust Pandas display options
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print("Descriptive Statistics for All Features:")
            print(stats_df)
        
    def _plot_descriptive_stats(self, progress_log):
        """
        Generate plots for features using the progress log.
        
        Args:
            progress_log (dict): The progress log containing stats and bin frequencies.
        """
        for feature, stats in progress_log["final_stats"].items():
            bin_edges = progress_log["bin_edges"][feature]
            bin_frequencies = progress_log["bin_frequencies"][feature]
    
            # Extract mean and standard deviation from final stats
            mean = stats["mean"]
            std_dev = stats["std"]
    
            # Define 3-sigma range
            lower_bound = mean - 4 * std_dev
            upper_bound = mean + 4 * std_dev
    
            # Identify bins inside the 3σ range
            valid_bins = [i for i, edge in enumerate(bin_edges[:-1]) if lower_bound <= edge <= upper_bound]
    
            # Extract valid bin frequencies
            frequencies = [bin_frequencies.get(i, 0) for i in valid_bins]
    
            # Add underflow and overflow bins
            underflow_count = sum(
                bin_frequencies[i] for i, edge in enumerate(bin_edges[:-1]) if edge < lower_bound
            )
            overflow_count = sum(
                bin_frequencies[i] for i, edge in enumerate(bin_edges[:-1]) if edge > upper_bound
            )
    
            # Add special bins for underflow and overflow
            frequencies.insert(0, underflow_count)
            frequencies.append(overflow_count)
            labels = ["< 4σ"] + [f"{bin_edges[i]:.2f}" for i in valid_bins] + ["> 4σ"]
    
            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(frequencies)), frequencies, width=0.9, align="center", alpha=0.7)
            plt.title(f"Feature: {feature} - Bin Distribution with Overflow & Underflow")
            plt.xlabel("Bins")
            plt.ylabel("Frequency")
            plt.xticks(range(len(frequencies)), labels, rotation=45, ha="right")
            plt.tight_layout()
            plt.show() 
            
    def _plot_rates(self, data, plot_skip):
        """
        Plot candlestick charts from OHLC standardized data using mplfinance.
    
        Steps:
        1. Identify OHLC-standardized features and ensure they form complete sets.
        2. Verify that all unique numbers (time steps) are sequential.
        3. Reconstruct each row into a time series DataFrame for plotting.
        4. Use a dummy date range to avoid NaT errors.
        5. Use `pair` and `date` for the plot title.
        6. Plot the full candlestick sequence from highest to lowest standard.
    
        Args:
            data (DataFrame): Data containing `pair`, `date`, and OHLC standardized columns.
        """
        # Ensure required columns exist
        if "date" not in data.columns or "pair" not in data.columns:
            print("Missing 'date' or 'pair' column in dataset.")
            return
    
        # Identify OHLC feature sets based on standardized naming pattern
        ohlc_pattern = re.compile(r"^(open|high|low|close)_standard_(\d+)$")
        feature_groups = {}
    
        for col in data.columns:
            match = ohlc_pattern.match(col)
            if match:
                ohlc_type, num = match.groups()
                num = int(num)
                if num not in feature_groups:
                    feature_groups[num] = {"open": None, "high": None, "low": None, "close": None}
                feature_groups[num][ohlc_type] = col  # Store column names under OHLC category
    
        # Ensure each group contains all four OHLC components
        valid_groups = {num: cols for num, cols in feature_groups.items() if all(cols.values())}
        
        if not valid_groups:
            print("Missing OHLC components in the dataset.")
            return
        
        # Ensure numbers are sequential
        sorted_numbers = sorted(valid_groups.keys(), reverse=True)
        if sorted_numbers != list(range(sorted_numbers[0], sorted_numbers[-1] - 1, -1)):
            print("Numbers are not sequential.")
            return
    
        # Process each row as a full candlestick sequence
        for index in range(0, len(data), plot_skip):
            row = data.iloc[index]
            pair = row["pair"]  # Extract currency pair name
            date_str = row["date"]  # Extract date as string
    
            # Construct DataFrame for mplfinance
            ohlc_data = pd.DataFrame({
                "Open": [row[valid_groups[num]["open"]] for num in sorted_numbers],
                "High": [row[valid_groups[num]["high"]] for num in sorted_numbers],
                "Low": [row[valid_groups[num]["low"]] for num in sorted_numbers],
                "Close": [row[valid_groups[num]["close"]] for num in sorted_numbers],
            })
            
            # Ensure data is numeric
            ohlc_data = ohlc_data.apply(pd.to_numeric, errors="coerce")
            
            # Drop rows with all NaN values in OHLC columns
            if ohlc_data.isnull().all(axis=None):
                continue
    
            # Generate a dummy datetime index (starting at 2024-01-01)
            dummy_start_date = pd.Timestamp("2024-01-01")
            date_range = pd.date_range(start=dummy_start_date, periods=len(sorted_numbers), freq='H')  # Hourly intervals
            ohlc_data.index = date_range  
    
            # Plot using mplfinance
            fig, ax = plt.subplots(figsize=(10, 6))
            mpf.plot(ohlc_data, type="candle", style="charles", ax=ax)
            ax.set_title(f"Candlestick Chart for {pair} - {date_str}")
            plt.show()
        
    def _describe_features(self, key, chunk_key, data, progress_log, bin_edges, feature_list, finish):
        """
        Calculate and store statistics for features in the current data chunk.
        Aggregate statistics for accurate reporting across chunks.
    
        Args:
            key (str): Dataset key.
            chunk_key (str): Current chunk identifier.
            data (DataFrame): Data chunk.
            progress_log (dict): Progress log for the dataset.
            bin_edges (dict): Pre-initialized bin edges for all features.
            feature_list (list): List of features to process.
            finish (bool): Whether this is the last chunk.
        """
        chunk_stats = data.describe().T
    
        for feature in feature_list:
            if feature not in progress_log["running_sum"]:
                # Initialize progress log for this feature
                progress_log["running_sum"][feature] = 0
                progress_log["running_square_sum"][feature] = 0
                progress_log["running_count"][feature] = 0
                progress_log["running_min"][feature] = float("inf")
                progress_log["running_max"][feature] = float("-inf")
                progress_log["bin_frequencies"][feature] = {i: 0 for i in range(len(bin_edges[feature]) - 1)}
                progress_log["bin_frequencies"][feature].update({"underflow": 0, "overflow": 0})
                progress_log["bin_edges"][feature] = bin_edges[feature]
    
            if feature in chunk_stats.index:
                # Update stats
                progress_log["running_sum"][feature] += chunk_stats.loc[feature, "mean"] * chunk_stats.loc[feature, "count"]
                progress_log["running_square_sum"][feature] += (
                    chunk_stats.loc[feature, "std"] ** 2 + chunk_stats.loc[feature, "mean"] ** 2
                ) * chunk_stats.loc[feature, "count"]
                progress_log["running_count"][feature] += chunk_stats.loc[feature, "count"]
                progress_log["running_min"][feature] = min(progress_log["running_min"][feature], chunk_stats.loc[feature, "min"])
                progress_log["running_max"][feature] = max(progress_log["running_max"][feature], chunk_stats.loc[feature, "max"])
    
                # Update bin frequencies
                self._update_bin_frequencies(
                    feature, data, bin_edges[feature], progress_log["bin_frequencies"]
                )
    
        if finish:
            self._finalize_describe_report(key, progress_log)
            self._display_descriptive_stats(progress_log)
    
    def _describe_targets(self, data_keys):
        """
        """
        
    def _plot_features(self, key, chunk_key, data, progress_log, bin_edges, 
                       feature_list, finish, describe_features):
        """
        Calculate and store statistics for features in the current data chunk.
        Aggregate statistics for accurate plotting across chunks.
    
        Args:
            key (str): Dataset key.
            chunk_key (str): Current chunk identifier.
            data (DataFrame): Data chunk.
            progress_log (dict): Progress log for the dataset.
            bin_edges (dict): Pre-initialized bin edges for all features.
            feature_list (list): List of features to process.
            finish (bool): Whether this is the last chunk.
            describe_features (bool): Is progress_log being created elsewhere.
        """
        # If describe_features is True, it will handle progress_log creation
        if describe_features:
            if finish:
                self._plot_descriptive_stats(progress_log)
        # If describe_features is False, create progress_log
        else:
            chunk_stats = data.describe().T
        
            for feature in feature_list:
                if feature not in progress_log["running_sum"]:
                    # Initialize progress log for this feature
                    progress_log["running_sum"][feature] = 0
                    progress_log["running_square_sum"][feature] = 0
                    progress_log["running_count"][feature] = 0
                    progress_log["running_min"][feature] = float("inf")
                    progress_log["running_max"][feature] = float("-inf")
                    progress_log["bin_frequencies"][feature] = {i: 0 for i in range(len(bin_edges[feature]) - 1)}
                    progress_log["bin_frequencies"][feature].update({"underflow": 0, "overflow": 0})
                    progress_log["bin_edges"][feature] = bin_edges[feature]
        
                if feature in chunk_stats.index:
                    # Update stats
                    progress_log["running_sum"][feature] += chunk_stats.loc[feature, "mean"] * chunk_stats.loc[feature, "count"]
                    progress_log["running_square_sum"][feature] += (
                        chunk_stats.loc[feature, "std"] ** 2 + chunk_stats.loc[feature, "mean"] ** 2
                    ) * chunk_stats.loc[feature, "count"]
                    progress_log["running_count"][feature] += chunk_stats.loc[feature, "count"]
                    progress_log["running_min"][feature] = min(progress_log["running_min"][feature], chunk_stats.loc[feature, "min"])
                    progress_log["running_max"][feature] = max(progress_log["running_max"][feature], chunk_stats.loc[feature, "max"])
        
                    # Update bin frequencies
                    self._update_bin_frequencies(
                        feature, data, bin_edges[feature], progress_log["bin_frequencies"]
                    )
        
            if finish:
                self._finalize_describe_report(key, progress_log)
                self._plot_descriptive_stats(progress_log)
        
    def _clean(self):
        """
        Delete features and targets from the original databse using the filter
        function.
        """
        
    def _align(self):
        """
        """
        
    def _collect_features(self, key):
        """
        Collect the feature (column) names from the source query for a given key.
    
        Args:
            key (str): The key identifying the dataset.
    
        Returns:
            list: A list of column names to process, excluding the primary key.
        """
        # Extract the query string for the given key
        query = self.source_query[key][1]
    
        # Parse the query to collect column headers
        # Extract column names between `SELECT` and `FROM`
        select_section = query.split("FROM")[0].split("SELECT")[-1].strip()
        columns = [col.strip() for col in select_section.split(",")]
    
        # Check for wildcard (*), fetch all columns if present
        if "*" in columns:
            # Establish a connection to the database
            database_name = self.source_query[key][0]
            db_path = self.env_loader.get(database_name)
            table_name = query.split("FROM")[1].split("WHERE")[0].strip()
    
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    query_for_columns = f"PRAGMA table_info({table_name})"
                    column_info = cursor.execute(query_for_columns).fetchall()
                    columns = [col[1] for col in column_info]  # Assuming column names are in the second position
            except Exception as e:
                raise RuntimeError(f"Failed to fetch column information: {e}")
    
        # Remove primary key columns
        dummy_df = pd.DataFrame(columns=columns)  # Create a dummy DataFrame with the column names
        filtered_df = self._remove_primary_key(dummy_df)
        feature_list = filtered_df.columns.tolist()
        return feature_list
        
    def _remove_primary_key(self, data):
        """
        Remove the primary key from the dataframe.
        """
        if not hasattr(self, "primary_key") or not self.primary_key:
            raise AttributeError("Primary key is not defined or empty.")
    
        # Ensure primary_key columns exist in the dataframe before dropping
        missing_keys = [key for key in self.primary_key if key not in data.columns]
        if missing_keys:
            raise ValueError(f"Primary key columns not found in dataframe: {missing_keys}")
    
        # Drop the primary key columns
        return data.drop(columns=self.primary_key)
    
    def _convert_to_numeric(self, data):
        """
        Convert column data from text to numeric.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
    
        # Attempt to convert all columns to numeric
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            except Exception as e:
                print(f"Could not convert column '{col}' to numeric: {e}")
    
        return data
        
    @public_method
    def clean_data(self,
                   data_keys,
                   describe_features=False,
                   describe_targets=False,
                   plot_features=False,
                   plot_mode='rate',
                   plot_skip=48,
                   clean=False):
        """
        Loop through the data keys and apply the selected functions to each
        data set in chunks. Supports iterative evaluation and cleaning.
        
        plot_mode options:
            - 'rate' will plot standardized OHLC data
            - 'stat' will plot histogram of feature distributions
        """
        # Separate describe/plot phase from cleaning/aligning
        if any([describe_features, describe_targets, plot_features]):
            for key in data_keys:
                print(f"Beginning descriptive functions for {key}...")
                chunk_keys = self.ltd.chunk_keys(key)  # Determine chunk splits
    
                for idx, chunk_key in enumerate(chunk_keys):
                    print(f"Processing chunk {chunk_key[0][0]} - {chunk_key[1][0]}...")
                    # Set finish to True for the last chunk
                    finish = idx == len(chunk_keys) - 1
                    # Initialize bins only if needed
                    if idx == 0:
                        if not (plot_features and plot_mode == 'rate' and not any([describe_features, describe_targets])):
                            feature_list = self._collect_features(key)
                            bin_edges = self._initialize_bins_from_sql(feature_list, key)
                    
                    data = self.ltd.load_chunk(key, chunk_key)
                    if not (plot_features and plot_mode == 'rate' and not any([describe_features, describe_targets])):
                        data = self._remove_primary_key(data)
                        data = self._convert_to_numeric(data)
                        progress_log = self._get_progress_log(key)
                    
                    # Perform operations on the current chunk
                    if describe_features:
                        self._describe_features(
                            key, chunk_key, data, progress_log, bin_edges, 
                            feature_list, finish)
                    if describe_targets:
                        self._describe_targets(key, chunk_key, data, finish)
                    if plot_features:
                        if plot_mode == 'rate':
                            self._plot_rates(data, plot_skip)
                        elif plot_mode == 'stat':
                            self._plot_features(key, chunk_key, data, progress_log, 
                                                bin_edges, feature_list, finish, 
                                                describe_features)
        
        if clean:
            progress_log = self._get_progress_log(key)
            chunk_keys = self.ltd.chunk_keys(key)  # Determine chunk splits
    
            for idx, chunk_key in enumerate(chunk_keys):
                # Set finish to True for the last chunk
                finish = idx == len(chunk_keys) - 1
    
                data = self.ltd.load_chunk(key, chunk_key)
                bad_keys = self._clean(key, chunk_key, data, finish)
                if bad_keys:
                    progress_log["bad_keys"].extend(bad_keys)
    
            # Perform alignment after all chunks are processed
            self._align(data_keys, progress_log)
        
    @public_method
    def engineer(self, mode="all"):
        """
        Create new features or targets and add them to new database tables.
        
        Args:
            mode = Options: all, feature, or target.
        """
        
    @public_method
    def encode_targets(self):
        """
        Use encoding logic to transform targets into numerical categories
        or appropiately format regression-based targets.
        """
        
    def _save_metadata(self):
        """
        Keep track of primary-key based splits, completed processes, etc.
        """
        
    @public_method
    def save_batches(self):
        """
        Split, type features and targets, scale and reshape features, and
        save completely prepped data in batches to .np, .npy, or .hdf5 files
        for efficient, iterative loading in model training.
        """
        
    ### OLD FUNCTIONS
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
        
        # Set pandas options to display all columns
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_rows', None)     # Show all rows for descriptive statistics
        pd.set_option('display.width', 1000)       # Adjust the width to fit all columns

        for name, df in describe_dfs:
            print(f"\nDataFrame: {name} - Descriptive Statistics")
            df_columns = [col for col in self._get_df_columns(df, columns) 
                              if col not in self.primary_key]
            print(df[df_columns].describe(include="all"))
            
            # Reset pandas display options to default after processing
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
        
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
            samples = feature_arr.size // (timesteps * features)
    
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