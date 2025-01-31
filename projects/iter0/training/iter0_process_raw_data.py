# iter0_process_raw_data.py


import numpy as np
import pandas as pd


def filter_hourly(df):
    """
    Find rows in df['close_standard_1'] with values above 1.01, below 0.99, 
    or containing NaN values.

    Returns:
        List of tuples containing ('pair', 'date') for matching rows.
    """
    if 'pair' not in df.columns or 'date' not in df.columns or 'close_standard_1' not in df.columns:
        raise ValueError("DataFrame must contain 'pair', 'date', and 'close_standard_1' columns.")

    # Apply filtering condition (values above 1.01 or below 0.99)
    filtered_rows = df[(df['close_standard_1'] > 1.01) | (df['close_standard_1'] < 0.99)]
    
    # Find rows with NaN values in any column
    nan_rows = df[df.isna().any(axis=1)]  # Any column containing NaN

    # Extract (pair, date) tuples
    filtered_list = list(zip(filtered_rows['pair'], filtered_rows['date']))
    nan_list = list(zip(nan_rows['pair'], nan_rows['date']))

    # Combine both lists
    return filtered_list + nan_list


def filter_targets(df):
    """
    Find rows in df['hours_passed'] with values >= 126.
    
    Returns:
        List of tuples containing ('pair', 'date') for matching rows.
    """
    if 'pair' not in df.columns or 'date' not in df.columns or 'hours_passed' not in df.columns:
        raise ValueError("DataFrame must contain 'pair', 'date', and 'hours_passed' columns.")

    # Apply filtering condition
    filtered_rows = df[df['hours_passed'] >= 126]
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
    Perform feature engineering by:
        - Adding sine and cosine transformations for the day of the month.
        - One-hot encoding the 'pair' column while retaining it.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'pair' columns.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle invalid date conversions
    if df['date'].isna().any():
        raise ValueError("Some dates could not be converted to datetime. Check 'date' column.")

    # Day-of-month cyclic encoding
    df['day_sin'] = np.sin(2 * np.pi * (df['date'].dt.day / 30))
    df['day_cos'] = np.cos(2 * np.pi * (df['date'].dt.day / 30))

    # One-hot encode 'pair' column while keeping the original
    df_encoded = pd.get_dummies(df, columns=['pair'], prefix='pair', dtype=int)
    
    # Re-insert the original 'pair' column at its original position
    df_encoded.insert(df.columns.get_loc('pair'), 'pair', df['pair'])

    return df_encoded


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
    