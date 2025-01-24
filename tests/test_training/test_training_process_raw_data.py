# test_training_process_raw_data.py


'''
Template of how to prepare low level modules for ProcessRawData. Some of these
functions may require multiple iterations when dealing with multiple DataFrames.
'''

import numpy as np
import pandas as pd


def required(func):
    func.required = True
    return func

def optional(func):
    func.optional = True
    return func


@optional
def filter_indices(df):
    """
    Identify problematic indices based on various conditions in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Index: Indices of rows that match problematic conditions.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Define conditions for specific features if they exist
    condition_1 = numeric_df['feature1'].eq(0) if 'feature1' in numeric_df.columns else pd.Series(False, index=numeric_df.index)
    condition_2 = numeric_df['feature2'].eq(0) if 'feature2' in numeric_df.columns else pd.Series(False, index=numeric_df.index)
    condition_3 = numeric_df['feature3'].eq(0) if 'feature3' in numeric_df.columns else pd.Series(False, index=numeric_df.index)

    # Flag rows where all three conditions are met
    combined_feature_condition = condition_1 & condition_2 & condition_3

    # Check for problematic values (NaN, inf, large numbers)
    condition_na = numeric_df.isna().any(axis=1)  # Rows with NaN
    condition_inf = numeric_df.isin([np.inf, -np.inf]).any(axis=1)  # Rows with inf/-inf
    condition_large = (numeric_df.abs() > np.finfo(np.float32).max).any(axis=1)  # Unreasonably large values

    # Combine all conditions
    problematic_indices = numeric_df.index[
        combined_feature_condition | condition_na | condition_inf | condition_large
    ]

    return problematic_indices

@optional
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

@optional
def target_engineering(df, mode='to_cat'):
    """
    Example function that converts encoded target column into string categories.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.
        mode (str): 'to_cat' or 'to_cont'

    Returns:
        pd.DataFrame: DataFrame with new targets.
        """
    if 'target' in df.columns:
        if mode == 'to_cont':
            df['target'] = np.where(
                df['target'] == 1,
                np.random.normal(loc=1, scale=2, size=len(df)),
                np.random.normal(loc=3, scale=1, size=len(df))
            )
        elif mode == 'to_cat':
            df['target'] = np.where(
                df['target'] == 1,
                'buy',
                'sell'
                )
    
    return df

@required
def df_features(df):
    """
    Example function that returns the df's list of training features.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.
        
    Returns:
        list: List of feature columns.
    """
    return [col for col in df.columns if col not in ['date', 'asset', 'target']]
    