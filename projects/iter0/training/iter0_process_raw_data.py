# iter0_process_raw_data.py


import numpy as np
import pandas as pd
from functools import reduce
import operator


def filter_hourly(df):
    """
    Filters rows in the DataFrame `df` based on different thresholds for columns with names
    following the pattern <open/high/low/close>_standard_<i>.

    The thresholds are applied as follows:
      - For columns with indices 36 to 48: a value is flagged if it is above 1.03 or below 0.97.
      - For columns with indices 18 to 35: a value is flagged if it is above 1.05 or below 0.95.
      - For columns with indices 1 to 17: a value is flagged if it is above 1.07 or below 0.93.
    
    Additionally, rows that contain any NaN values (in any column) are also flagged.

    Finally, the function returns a list of tuples containing the ('pair', 'date') values
    for the rows that meet any of these conditions.

    Args:
        df (pandas.DataFrame): The DataFrame to filter. It must contain at least the columns 'pair' and 'date',
                               as well as the <open/high/low/close>_standard_<i> columns.

    Returns:
        List[Tuple]: A list of (pair, date) tuples for the rows that meet the filter conditions.
    """
    conditions = []
    prefixes = ['open', 'high', 'low', 'close']

    # For columns 36 to 48: ±3% threshold (values > 1.03 or < 0.97)
    for i in range(36, 49):  # 49 is exclusive
        for prefix in prefixes:
            col = f"{prefix}_standard_{i}"
            if col in df.columns:
                cond = (df[col] > 1.03) | (df[col] < 0.97)
                conditions.append(cond)

    # For columns 18 to 35: ±5% threshold (values > 1.05 or < 0.95)
    for i in range(18, 36):
        for prefix in prefixes:
            col = f"{prefix}_standard_{i}"
            if col in df.columns:
                cond = (df[col] > 1.05) | (df[col] < 0.95)
                conditions.append(cond)

    # For columns 1 to 17: ±7% threshold (values > 1.07 or < 0.93)
    for i in range(1, 18):
        for prefix in prefixes:
            col = f"{prefix}_standard_{i}"
            if col in df.columns:
                cond = (df[col] > 1.07) | (df[col] < 0.93)
                conditions.append(cond)

    # Combine all the conditions with bitwise OR; if no condition exists, use a default (all False)
    if conditions:
        combined_condition = reduce(operator.or_, conditions)
    else:
        combined_condition = pd.Series(False, index=df.index)

    # Filter rows based on the combined condition.
    filtered_rows = df[combined_condition]

    # Find rows with NaN values in any column.
    nan_rows = df[df.isna().any(axis=1)]

    # Combine both DataFrames. This avoids duplicates if some rows meet both conditions.
    combined = pd.concat([filtered_rows, nan_rows]).drop_duplicates()

    # Extract (pair, date) tuples.
    return list(zip(combined['pair'], combined['date']))


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
    # Original encoding, uncomment if engineering data from original target table
    #category_to_index = {'loss': 0, 'buy': 1, 'sell': 2, 'wait': 0}    
    #df['target'] = df['target'].map(category_to_index)
    
    df['date'] = pd.to_datetime(df['date'])
    date_boundary = pd.to_datetime('2022-11-30')
    df['hour'] = df['date'].dt.hour
    condition_before_date = (df['date'] < date_boundary) & (df['hour'].isin([20, 21, 22]))
    condition_after_date = (df['date'] > date_boundary) & (df['hour'].isin([8, 9, 10]))
    df['hours_passed'] = df['hours_passed'].fillna(0).astype(float)
    condition_hours_passed = df['hours_passed'] >= 6
    
    df['target'] = np.where(
        condition_before_date | condition_after_date | condition_hours_passed,
        0,
        df['target']
        )
    
    df.drop(['hour'], axis=1, inplace=True)
    
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
    