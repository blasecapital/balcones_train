# load_training_data.py


import sqlite3
import pandas as pd
import math

from utils import EnvLoader, public_method


class LoadTrainingData:
    def __init__(self):
        """
        Initialize the LoadData class with configuration and utility modules.
        """
        # Initialize the EnvLoader
        self.env_loader = EnvLoader()
        
        # Retrieve the path to the config module using the env_loader
        config_path = self.env_loader.get("DATA_TRAIN_CONFIG")
        
        # Dynamically import the config module
        self.config = self.env_loader.load_config_module(config_path)
        
        self.source_query = self.config.get('source_query')
        self.primary_key = self.config.get('primary_key')
        self.max_chunk_size = 20_000_000  # Limit on rows * columns per chunk
        
    def _get_row_count(self, db_path, query):
        """
        Get the number of rows the query will return.

        Args:
            db_path (str): Path to the SQLite database.
            query (str): SQL query.

        Returns:
            int: Number of rows the query will return.
        """
        cleaned_query = query.strip().rstrip(";")
        count_query = f"SELECT COUNT(*) FROM ({cleaned_query}) as temp_table"
        with sqlite3.connect(db_path) as conn:
            row_count = conn.execute(count_query).fetchone()[0]
        return row_count

    def _get_column_count(self, db_path, query):
        """
        Get the number of columns in the query result.

        Args:
            db_path (str): Path to the SQLite database.
            query (str): SQL query.

        Returns:
            int: Number of columns in the query result.
        """
        cleaned_query = query.strip().rstrip(";")
        limit_query = f"SELECT * FROM ({cleaned_query}) as temp_table LIMIT 1"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(limit_query)
            column_count = len(cursor.description)  # Number of columns
        return column_count

    def chunk_keys(self, key):
        """
        Calculate primary key ranges for chunks based on the dataset size
        and the maximum allowable chunk size. Supports composite primary keys.
    
        Args:
            key (str): Key from the source_query config.
    
        Returns:
            list of tuple: List of ((start_date, start_pair), (end_date, end_pair)) tuples for chunking.
        """
        if key not in self.source_query:
            raise ValueError(f"Key '{key}' not found in source_query configuration.")
        
        db_ref, query = self.source_query[key]
        db_path = self.env_loader.get(db_ref)  # Retrieve the database path from EnvLoader
        
        # Get row and column counts
        row_count = self._get_row_count(db_path, query)
        column_count = self._get_column_count(db_path, query)
    
        # Calculate the initial chunk size
        max_rows_per_chunk = self.max_chunk_size // column_count
        split_factor = 5
    
        while row_count // split_factor > max_rows_per_chunk:
            split_factor += 5
    
        # Calculate the number of rows per chunk
        rows_per_chunk = math.ceil(row_count / split_factor)
    
        # Prepare the primary key and ordering logic
        primary_key_columns = ", ".join(self.primary_key)
        order_by_clause = ", ".join(self.primary_key)
    
        # Extract filtering logic from the source query
        base_query = query.strip().rstrip(";")
        select_primary_keys = f"""
            SELECT DISTINCT {primary_key_columns}
            FROM ({base_query})
            ORDER BY {order_by_clause}
        """
    
        # Retrieve all distinct primary key tuples for splitting
        with sqlite3.connect(db_path) as conn:
            primary_key_values = [tuple(row) for row in conn.execute(select_primary_keys)]
    
        # Split the primary key values into chunks
        chunk_keys = []
        for i in range(0, len(primary_key_values), rows_per_chunk):
            chunk_keys.append(
                (primary_key_values[i], 
                 primary_key_values[min(i + rows_per_chunk - 1, len(primary_key_values) - 1)]))
    
        return chunk_keys

    def load_chunk(self, key, chunk_key):
        """
        Load a chunk of data based on the key and chunk key.
    
        Args:
            key (str): Key from the source_query config.
            chunk_key (tuple): Tuple of (start_date, end_date) for chunk boundaries.
    
        Returns:
            pandas.DataFrame: DataFrame containing the data for the chunk.
        """
        if key not in self.source_query:
            raise ValueError(f"Key '{key}' not found in source_query configuration.")
        
        # Extract database and query information
        db_ref, base_query = self.source_query[key]
        db_path = self.env_loader.get(db_ref)  # Retrieve the database path from EnvLoader
        
        # Parse the base query to extract WHERE filtering logic
        base_query = base_query.strip().rstrip(";")
        if "WHERE" in base_query.upper():
            where_clause_start = base_query.upper().index("WHERE")
            select_part = base_query[:where_clause_start].strip()
            where_part = base_query[where_clause_start:].strip()
        else:
            select_part = base_query.strip()
            where_part = ""  # No filtering logic present
    
        # Extract the column list or use "*" as default
        if "SELECT" in select_part.upper() and "FROM" in select_part.upper():
            column_part = select_part.split("FROM")[0].replace("SELECT", "").strip()
        else:
            column_part = "*"  # Fallback to select all columns
    
        # Extract table name from the base query
        table_name = select_part.split("FROM")[1].strip()
    
        # Build the final query with the chunk key's date range
        where_clause = f"""
            {where_part} {"AND" if where_part else "WHERE"} 
            date > ? AND date <= ?
        """
        final_query = f"""
            SELECT {column_part}
            FROM {table_name}
            {where_clause}
        """
    
        # Extract start_date and end_date, ensure they are strings
        start_date = str(chunk_key[0][0])
        end_date = str(chunk_key[1][0])
        with sqlite3.connect(db_path) as conn:
            data = pd.read_sql_query(final_query, conn, params=(start_date, end_date))
    
        return data
    
    ### OLD #########
    @public_method
    def load_mode(self, db_path, query):
        """
        Check the shape of the data to load and determine if chunking is necessary.

        Args:
            db_path (str): Path to the SQLite database.
            query (str): SQL query.

        Returns:
            str: "chunk" if chunked loading is necessary, otherwise "full".
        """
        # Get the number of rows and columns
        row_count = self._get_row_count(db_path, query)
        column_count = self._get_column_count(db_path, query)

        # Estimate memory usage in GB (assuming 4 bytes per value)
        estimated_memory_gb = (row_count * column_count * 8) / (1024**3)

        print(f"Estimated DataFrame size: {row_count} rows x {column_count} columns")
        print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB")

        # Determine the load mode
        if row_count > self.chunk_size_threshold or estimated_memory_gb > self.memory_threshold_gb:
            print("Switching to chunked loading mode.")
            return "chunk"
        else:
            print("Full loading mode is sufficient.")
            return "full"
        
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
            