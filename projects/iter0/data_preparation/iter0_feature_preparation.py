# iter0_data_preparation.py


import pandas as pd
import numpy as np


#########################################################
# Functions for standardizing hourly exchange rates
# 
# Rates at 'ohlc'_standard_1 include the hour's rates,
# targets will need to be calculated starting from the
# following hour and its adjusted open price
#########################################################


def standardize_open_hourly_rates(df, shift=14):
    """
    Standardizes the Open values over the past 'shift' periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'open_standard_{hour + 1}': df['open'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'open_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_high_hourly_rates(df, shift=14):
    """
    Standardizes the High values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'high_standard_{hour + 1}': df['high'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'high_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_low_hourly_rates(df, shift=14):
    """
    Standardizes the Low values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'low_standard_{hour + 1}': df['low'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'low_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


def standardize_close_hourly_rates(df, shift=14):
    """
    Standardizes the Close values over the past 14 periods.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the OHLC data.
        start_index (int): Index from where to start the calculation.
        shift (int): Defines the shift from the current hour.
    
    Returns:
        pd.DataFrame: Updated DataFrame with standardized OHLC columns.
    """    
    # Ensure that the dependent columns are numeric and contain no NaNs
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Create all shifted columns at once
    shifted_data = {
        f'close_standard_{hour + 1}': df['close'].shift(hour)
        for hour in range(shift)
    }
    shifted_data['baseline_close'] = df['close'].shift(shift)
    
    # Add shifted columns to DataFrame in one operation to avoid fragmentation
    shifted_df = pd.DataFrame(shifted_data)
    
    # Calculate standardized values in one step
    for hour in range(shift):
        column_name = f'close_standard_{hour + 1}'
        shifted_df[column_name] = (
            ((shifted_df[column_name] - shifted_df['baseline_close']) /
             shifted_df['baseline_close']).round(5) + 1
        )
    
    # Merge back into the original DataFrame
    return pd.concat([df, shifted_df.drop(columns=['baseline_close'])], axis=1)


#########################################################
# Required functions for the data_preparation.py module
#########################################################


def features(df):
    """
    Call all feature functions and add them to the df.

    Args:
        df (pd.DataFrame): DataFrame containing exchange rates and other features.

    Returns:
        pd.DataFrame: Updated DataFrame with new features added for each pair.
    """
    df = standardize_open_hourly_rates(df, shift=48)
    df = standardize_high_hourly_rates(df, shift=48)
    df = standardize_low_hourly_rates(df, shift=48)
    df = standardize_close_hourly_rates(df, shift=48)
    return df

storage_map = {
    # Key: table name
    # Values: list of feature columns
    'hourly_features': ['open_standard_48', 'high_standard_48', 
                        'low_standard_48', 'close_standard_48', 
                        'open_standard_47', 'high_standard_47', 
                        'low_standard_47', 'close_standard_47', 
                        'open_standard_46', 'high_standard_46', 
                        'low_standard_46', 'close_standard_46', 
                        'open_standard_45', 'high_standard_45', 
                        'low_standard_45', 'close_standard_45', 
                        'open_standard_44', 'high_standard_44', 
                        'low_standard_44', 'close_standard_44', 
                        'open_standard_43', 'high_standard_43', 
                        'low_standard_43', 'close_standard_43', 
                        'open_standard_42', 'high_standard_42', 
                        'low_standard_42', 'close_standard_42', 
                        'open_standard_41', 'high_standard_41', 
                        'low_standard_41', 'close_standard_41', 
                        'open_standard_40', 'high_standard_40', 
                        'low_standard_40', 'close_standard_40', 
                        'open_standard_39', 'high_standard_39', 
                        'low_standard_39', 'close_standard_39', 
                        'open_standard_38', 'high_standard_38', 
                        'low_standard_38', 'close_standard_38', 
                        'open_standard_37', 'high_standard_37', 
                        'low_standard_37', 'close_standard_37', 
                        'open_standard_36', 'high_standard_36', 
                        'low_standard_36', 'close_standard_36', 
                        'open_standard_35', 'high_standard_35', 
                        'low_standard_35', 'close_standard_35', 
                        'open_standard_34', 'high_standard_34', 
                        'low_standard_34', 'close_standard_34', 
                        'open_standard_33', 'high_standard_33', 
                        'low_standard_33', 'close_standard_33', 
                        'open_standard_32', 'high_standard_32', 
                        'low_standard_32', 'close_standard_32', 
                        'open_standard_31', 'high_standard_31', 
                        'low_standard_31', 'close_standard_31', 
                        'open_standard_30', 'high_standard_30', 
                        'low_standard_30', 'close_standard_30', 
                        'open_standard_29', 'high_standard_29', 
                        'low_standard_29', 'close_standard_29', 
                        'open_standard_28', 'high_standard_28', 
                        'low_standard_28', 'close_standard_28', 
                        'open_standard_27', 'high_standard_27', 
                        'low_standard_27', 'close_standard_27', 
                        'open_standard_26', 'high_standard_26', 
                        'low_standard_26', 'close_standard_26', 
                        'open_standard_25', 'high_standard_25', 
                        'low_standard_25', 'close_standard_25', 
                        'open_standard_24', 'high_standard_24', 
                        'low_standard_24', 'close_standard_24', 
                        'open_standard_23', 'high_standard_23', 
                        'low_standard_23', 'close_standard_23', 
                        'open_standard_22', 'high_standard_22', 
                        'low_standard_22', 'close_standard_22', 
                        'open_standard_21', 'high_standard_21', 
                        'low_standard_21', 'close_standard_21', 
                        'open_standard_20', 'high_standard_20', 
                        'low_standard_20', 'close_standard_20', 
                        'open_standard_19', 'high_standard_19', 
                        'low_standard_19', 'close_standard_19', 
                        'open_standard_18', 'high_standard_18', 
                        'low_standard_18', 'close_standard_18', 
                        'open_standard_17', 'high_standard_17', 
                        'low_standard_17', 'close_standard_17', 
                        'open_standard_16', 'high_standard_16', 
                        'low_standard_16', 'close_standard_16', 
                        'open_standard_15', 'high_standard_15', 
                        'low_standard_15', 'close_standard_15', 
                        'open_standard_14', 'high_standard_14', 
                        'low_standard_14', 'close_standard_14', 
                        'open_standard_13', 'high_standard_13', 
                        'low_standard_13', 'close_standard_13', 
                        'open_standard_12', 'high_standard_12', 
                        'low_standard_12', 'close_standard_12', 
                        'open_standard_11', 'high_standard_11', 
                        'low_standard_11', 'close_standard_11', 
                        'open_standard_10', 'high_standard_10', 
                        'low_standard_10', 'close_standard_10', 
                        'open_standard_9', 'high_standard_9', 
                        'low_standard_9', 'close_standard_9', 
                        'open_standard_8', 'high_standard_8', 
                        'low_standard_8', 'close_standard_8', 
                        'open_standard_7', 'high_standard_7', 
                        'low_standard_7', 'close_standard_7', 
                        'open_standard_6', 'high_standard_6', 
                        'low_standard_6', 'close_standard_6', 
                        'open_standard_5', 'high_standard_5', 
                        'low_standard_5', 'close_standard_5', 
                        'open_standard_4', 'high_standard_4', 
                        'low_standard_4', 'close_standard_4', 
                        'open_standard_3', 'high_standard_3', 
                        'low_standard_3', 'close_standard_3', 
                        'open_standard_2', 'high_standard_2', 
                        'low_standard_2', 'close_standard_2', 
                        'open_standard_1', 'high_standard_1', 
                        'low_standard_1', 'close_standard_1']
    }
