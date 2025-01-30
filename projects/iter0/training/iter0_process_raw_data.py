# iter0_process_raw_data.py


import numpy as np
import pandas as pd


def filter_hourly(df):
    """
    Find rows in df['close_standard_1'] with values above 1.01 or below 0.99.
    
    Returns:
        List of tuples containing ('pair', 'date') for matching rows.
    """
    if 'pair' not in df.columns or 'date' not in df.columns or 'close_standard_1' not in df.columns:
        raise ValueError("DataFrame must contain 'pair', 'date', and 'close_standard_1' columns.")

    # Apply filtering condition
    filtered_rows = df[(df['close_standard_1'] > 1.01) | (df['close_standard_1'] < 0.99)]
    # Return as a list of tuples (pair, date)
    return list(zip(filtered_rows['pair'], filtered_rows['date']))
    

def filter_indices(df):
    """
    Identify problematic indices based on various conditions in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Index: Indices of rows that match problematic conditions.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    # Check for problematic values (NaN, inf, large numbers)
    condition_na = numeric_df.isna().any(axis=1)  # Rows with NaN
    condition_inf = numeric_df.isin([np.inf, -np.inf]).any(axis=1)  # Rows with inf/-inf
    condition_large = (numeric_df.abs() > np.finfo(np.float32).max).any(axis=1)  # Unreasonably large values

    # Combine all conditions
    problematic_indices = numeric_df.index[
        condition_na | condition_inf | condition_large
    ]

    return problematic_indices


def feature_engineering(df):
    """
    Example function that performs feature engineering by adding sine and 
    cosine transformations for the day of the month.

    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle rows where 'date' conversion failed (optional)
    if df['date'].isna().any():
        raise ValueError(r"Some dates could not be converted to datetime."
                         " Check the 'date' column for invalid values.")

    # Add sine and cosine transformations for the day of the month
    df['day_sin'] = np.sin(2 * np.pi * (df['date'].dt.day / 30))
    df['day_cos'] = np.cos(2 * np.pi * (df['date'].dt.day / 30))

    return df


def target_engineering(df):
    """
    Encodes target column into string categories.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with new targets.
        """
    category_to_index = {'loss': 0, 'buy': 1, 'sell': 2, 'wait': 0}    
    
    df['target'] = df['target'].map(category_to_index)
    
    return df


def df_features(df):
    """
    Example function that returns the df's list of training features.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.
        
    Returns:
        list: List of feature columns.
    """
    return [col for col in df.columns if col not in [
        'date', 'pair', 'target', 'hours_passed', 'buy_sl_time', 'sell_sl_time']]
    