# iter0_target_preparation.py


import re
from datetime import datetime, time
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


multipliers = {
    'AUD/CAD': 0.000252395,
    'AUD/CHF': 0.000564364,
    'AUD/CNH': 0.000397324,
    'AUD/JPY': 0.00024565,
    'AUD/NOK': 0.00055346,
    'AUD/NZD': 0.000442307,
    'AUD/PLN': 0.000832069,
    'AUD/SGD': 0.000953541,
    'AUD/USD': 0.000280151,
    'CAD/CHF': 0.000522715,
    'CAD/JPY': 0.000325966,
    'CAD/NOK': 0.000380918,
    'CAD/PLN': 0.00086648,
    'CHF/HUF': 0.0012863,
    'CHF/JPY': 0.000205996,
    'CHF/NOK': 0.000449401,
    'CHF/PLN': 0.000730289,
    'CNH/JPY': 0.000645206,
    'EUR/AUD': 0.000267104,
    'EUR/CAD': 0.00021782,
    'EUR/CHF': 0.000282483,
    'EUR/CNH': 0.000256059,
    'EUR/CZK': 0.001242339,
    'EUR/DKK': 6.57315E-05,
    'EUR/GBP': 0.000186624,
    'EUR/HKD': 0.000188408,
    'EUR/HUF': 0.000614183,
    'EUR/JPY': 0.000152115,
    'EUR/MXN': 0.001011392,
    'EUR/NOK': 0.000412265,
    'EUR/NZD': 0.000470346,
    'EUR/PLN': 0.000348448,
    'EUR/SEK': 0.000373188,
    'EUR/SGD': 0.000477034,
    'EUR/TRY': 0.002148228,
    'EUR/USD': 0.000110819,
    'EUR/ZAR': 0.000463247,
    'GBP/AUD': 0.000305286,
    'GBP/CAD': 0.000291714,
    'GBP/CHF': 0.000367865,
    'GBP/DKK': 0.000552219,
    'GBP/HKD': 0.000272526,
    'GBP/JPY': 0.000211919,
    'GBP/MXN': 0.00104095,
    'GBP/NOK': 0.000813789,
    'GBP/NZD': 0.000314019,
    'GBP/PLN': 0.001066847,
    'GBP/SEK': 0.000562136,
    'GBP/SGD': 0.000566698,
    'GBP/USD': 0.000118725,
    'GBP/ZAR': 0.000585705,
    'HKD/JPY': 0.000134595,
    'NOK/DKK': 0.001475407,
    'NOK/JPY': 0.000660042,
    'NOK/SEK': 0.001722628,
    'NZD/CAD': 0.000871872,
    'NZD/CHF': 0.000976488,
    'NZD/JPY': 0.00038264,
    'NZD/USD': 0.000540595,
    'SGD/HKD': 0.000674087,
    'SGD/JPY': 0.000446566,
    'TRY/JPY': 0.023159785,
    'USD/CAD': 0.000169543,
    'USD/CHF': 0.00023796,
    'USD/CNH': 0.00012078,
    'USD/CZK': 0.001076137,
    'USD/DKK': 0.000103147,
    'USD/HKD': 0.000170871,
    'USD/HUF': 0.000586157,
    'USD/ILS': 0.001398085,
    'USD/JPY': 9.60859E-05,
    'USD/MXN': 0.000283136,
    'USD/NOK': 0.000430461,
    'USD/PLN': 0.000428206,
    'USD/SEK': 0.000407746,
    'USD/SGD': 0.000428035,
    'USD/THB': 0.00117287,
    'USD/TRY': 0.002337786,
    'USD/ZAR': 0.000605328,
    'ZAR/JPY': 0.003099974
}


widemultipliers = {
    'AUD/CAD': 0.001226329,
    'AUD/CHF': 0.001513704,
    'AUD/CNH': 0.00060006,
    'AUD/JPY': 0.000679733,
    'AUD/NOK': 0.00426708,
    'AUD/NZD': 0.001121527,
    'AUD/PLN': 0.00583835,
    'AUD/SGD': 0.001771131,
    'AUD/USD': 0.000650051,
    'CAD/CHF': 0.001726594,
    'CAD/JPY': 0.001528995,
    'CAD/NOK': 0.005419667,
    'CAD/PLN': 0.006877273,
    'CHF/HUF': 0.00525359,
    'CHF/JPY': 0.001063064,
    'CHF/NOK': 0.004166512,
    'CHF/PLN': 0.004779603,
    'CNH/JPY': 0.001260033,
    'EUR/AUD': 0.000724601,
    'EUR/CAD': 0.000772216,
    'EUR/CHF': 0.001110391,
    'EUR/CNH': 0.000346413,
    'EUR/CZK': 0.003474211,
    'EUR/DKK': 0.000201599,
    'EUR/GBP': 0.000559497,
    'EUR/HKD': 0.000971889,
    'EUR/HUF': 0.00222013,
    'EUR/JPY': 0.000340832,
    'EUR/MXN': 0.002471325,
    'EUR/NOK': 0.002584394,
    'EUR/NZD': 0.000705849,
    'EUR/PLN': 0.003837503,
    'EUR/SEK': 0.001823482,
    'EUR/SGD': 0.001797112,
    'EUR/TRY': 0.005531749,
    'EUR/USD': 0.000368475,
    'EUR/ZAR': 0.001672755,
    'GBP/AUD': 0.000675847,
    'GBP/CAD': 0.000689707,
    'GBP/CHF': 0.000992367,
    'GBP/DKK': 0.002440446,
    'GBP/HKD': 0.001803367,
    'GBP/JPY': 0.000786392,
    'GBP/MXN': 0.002209237,
    'GBP/NOK': 0.003415718,
    'GBP/NZD': 0.000630852,
    'GBP/PLN': 0.003944688,
    'GBP/SEK': 0.0035608,
    'GBP/SGD': 0.002669695,
    'GBP/USD': 0.0003147,
    'GBP/ZAR': 0.0022424,
    'HKD/JPY': 0.0017277,
    'NOK/DKK': 0.004417161,
    'NOK/JPY': 0.003327621,
    'NOK/SEK': 0.005649126,
    'NZD/CAD': 0.00131268,
    'NZD/CHF': 0.001888247,
    'NZD/JPY': 0.001672286,
    'NZD/USD': 0.000696166,
    'SGD/HKD': 0.00196306,
    'SGD/JPY': 0.002923452,
    'TRY/JPY': 0.018472727,
    'USD/CAD': 0.000392915,
    'USD/CHF': 0.00056508,
    'USD/CNH': 0.000527567,
    'USD/CZK': 0.003731517,
    'USD/DKK': 0.000246308,
    'USD/HKD': 0.000193494,
    'USD/HUF': 0.002712532,
    'USD/ILS': 0.004148938,
    'USD/JPY': 0.000455197,
    'USD/MXN': 0.002011915,
    'USD/NOK': 0.002931446,
    'USD/PLN': 0.002172391,
    'USD/SEK': 0.002307243,
    'USD/SGD': 0.000908073,
    'USD/THB': 0.004006678,
    'USD/TRY': 0.006766476,
    'USD/ZAR': 0.001852337,
    'ZAR/JPY': 0.012747064
}


def normalize_pair_name(pair_name):
    """
    Extracts and normalizes the currency pair name from a pair name.
    Normalizes formats like 'AUDUSD', 'aud/usd', 'AUD/usd' to 'AUD/USD'.
    """
    # Extract the base name (without path and extension)
    base_name = re.sub(r'[^a-zA-Z]', '', pair_name).upper()  # Remove non-alphabetic chars and convert to upper case
    
    # Assuming currency pairs are 6 characters long (e.g., 'AUDUSD')
    if len(base_name) == 6:
        return base_name[:3] + '/' + base_name[3:]
    else:
        raise ValueError("Currency pair name format not recognized.")
        
        
def get_multiplier(date, normalized_pair_name, multipliers, widemultipliers):
    """
    Determines the appropriate multiplier to use based on the timestamp.
    """
    date = pd.to_datetime(date)
    is_wide_multiplier = False
    if date.date() < datetime(2022, 11, 30).date() and date.time() == time(21, 0):
        factor = widemultipliers.get(normalized_pair_name, 0)
        is_wide_multiplier = True
    elif date.date() >= datetime(2022, 11, 30).date() and date.time() == time(9, 0):
        factor = widemultipliers.get(normalized_pair_name, 0)
        is_wide_multiplier = True
    else:
        factor = multipliers.get(normalized_pair_name, 0)
    
    return factor, is_wide_multiplier


def add_open_prices(df, pair_name, multipliers, widemultipliers):
    """
    """
    
    # Normalize the pair name based on the file name
    normalized_pair_name = normalize_pair_name(pair_name)
        
    def calculate_open_price_bid(row):
        open_price_bid = row['open']
        return open_price_bid

    def calculate_open_price_ask(row):
        date = pd.to_datetime(row['date']).tz_localize(None)
        factor, _ = get_multiplier(date, normalized_pair_name, multipliers, widemultipliers)
        adjustment_factor = 1 + factor
        open_price_bid = row['open_price_bid']
        open_price_ask = open_price_bid * adjustment_factor
        return open_price_ask

    df['open_price_bid'] = df.apply(calculate_open_price_bid, axis=1)
    df['open_price_ask'] = df.apply(calculate_open_price_ask, axis=1)

    return df


def calc_sl_tp(df, pair):
    """
    Calculate the stop loss and take profit based on recent volatility.
    
    Parameters:
    df (DataFrame): A pandas DataFrame with at least 'high', 'low', 
                    'open_price_ask', and 'open_price_bid' columns.
    pair (str): The currency pair to normalize and retrieve spread multipliers for.
    
    Returns:
    DataFrame: The DataFrame with new columns 'true_range', 'avg_true_range',
               'buy_sl', 'buy_tp', 'sell_sl', and 'sell_tp'.
    """
    # Calculate true range
    df['true_range'] = df['high'] - df['low']
    
    # Find the normal spread amount
    spread = multipliers[pair]
    
    # Calculate the rolling 14-period average of the true range
    df['avg_true_range'] = df['true_range'].rolling(window=14).mean() * (1 + spread)
    df['avg_true_range'] = df['avg_true_range'].fillna(0)
     
    # Calculate buy sl and tp
    df['buy_sl'] = df['open_price_ask'] - np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    df['buy_tp'] = df['open_price_ask'] + 2 * np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    
    # Calculate sell sl and tp
    df['sell_sl'] = df['open_price_bid'] + np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    df['sell_tp'] = df['open_price_bid'] - 2 * np.maximum(
        df['avg_true_range'], 
        df['open_price_ask'] * 0.00125)
    
    return df
    

def true_outcome(index, df, multipliers, widemultipliers, n):
    outcome = 'wait'  # Default target
    hours_passed = 0  # Initialize hours passed
    
    # Check if out of bounds
    if index + 1 >= len(df):
        return outcome, hours_passed, None, None
    
    row = df.iloc[index]
    start_row = df.iloc[index+1]
    pair_name = normalize_pair_name(row['pair'])
    date = pd.to_datetime(row['date'])
    
    # Get the multiplier for the pair
    multiplier, is_wide_multiplier = get_multiplier(
        date, pair_name, multipliers, widemultipliers)
    
    # Calculate the take profit and stop loss for 'Buy'
    buy_take_profit = start_row['buy_tp']
    buy_stop_loss = start_row['buy_sl']
    
    # Calculate the take profit and stop loss for 'Sell'
    sell_take_profit = start_row['sell_tp']
    sell_stop_loss = start_row['sell_sl']
    
    # Flags to track if stop loss for buy and sell are hit
    buy_stop_loss_hit = False
    sell_stop_loss_hit = False
    
    buy_sl_time = None
    sell_sl_time = None
    
    # Loop through the next set of hours
    for i in range(index, min(index + n, len(df) - 1)):
        next_row = df.iloc[i+1]
        current_timestamp = pd.to_datetime(next_row['date'])
        
        # Determine the appropriate multiplier based on the is_wide_multiplier flag
        effective_multiplier = 1 + multiplier if not is_wide_multiplier else 1 + is_wide_multiplier
        
        # Check for 'Buy' and 'Sell' stop loss conditions
        if next_row['low'] < buy_stop_loss and not buy_sl_time:
            buy_stop_loss_hit = True
            buy_sl_time = (current_timestamp - date).total_seconds() / 3600
        if next_row['high'] * effective_multiplier > sell_stop_loss and not sell_sl_time:
            sell_stop_loss_hit = True
            sell_sl_time = (current_timestamp - date).total_seconds() / 3600

        # If both stop losses are hit, assign 'Loss' and exit loop
        if buy_stop_loss_hit and sell_stop_loss_hit:
            outcome = 'loss'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break

        # Check for 'Buy' condition
        if not buy_stop_loss_hit and next_row['high'] >= buy_take_profit:
            outcome = 'buy'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break  # Take profit was hit, stop checking further
        
        # Check for 'Sell' condition
        if not sell_stop_loss_hit and next_row['low'] * effective_multiplier <= sell_take_profit:
            outcome = 'sell'
            hours_passed = (current_timestamp - date).total_seconds() / 3600
            break  # Take profit was hit, stop checking further
    
    return outcome, hours_passed, buy_sl_time, sell_sl_time


def update_targets(df, multipliers, widemultipliers, n=336):
    pair_name = normalize_pair_name(df['pair'][0])
    df = add_open_prices(df, pair_name, multipliers, widemultipliers)
    df = calc_sl_tp(df, pair_name)
    
    # Initialize an empty list to hold the targets
    outcomes = []
    hours_passed = []
    buy_sl = []
    sell_sl = []
    
    # Iterate over each row index in the DataFrame
    for index in tqdm(range(len(df)), desc="Determining targets"):
        # Call true_outcome for each row and append the result to the targets list
        outcome, hours, buy_sl_time, sell_sl_time = true_outcome(
            index, df, multipliers, widemultipliers, n)
        outcomes.append(outcome)
        hours_passed.append(hours)
        buy_sl.append(buy_sl_time)
        sell_sl.append(sell_sl_time)
    
    # Assign the list of targets back to the DataFrame's 'Target' column
    df['target'] = outcomes
    df['hours_passed'] = hours_passed
    df['buy_sl_time'] = buy_sl
    df['sell_sl_time'] = sell_sl
    
    output_df = df[['date', 'pair', 'target', 'hours_passed', 'buy_sl_time',
                    'sell_sl_time']]
    return output_df


#########################################################
# Required functions for the data_preparation.py module
#########################################################

#db_path = r"C:\Users\brand\OneDrive\Blase Capital Mgmt\deep_learning\projects\data\base.db"
'''
query = """
SELECT * FROM hourly_exchange_rates
WHERE pair = 'AUDUSD'
AND date > '2023-01-01'
"""

import sqlite3 as sql
conn = sql.connect(db_path)
try:
    df = pd.read_sql(query, conn)
    print(df)
    df = update_targets(df, multipliers, widemultipliers, n=336)
    print(df.columns)
    print(df[['date', 'outcome']])
    print(df['outcome'].value_counts())
finally:
    conn.close()
'''

def targets(df):
    """
    Call all feature functions and add them to the df.

    Args:
        df (pd.DataFrame): DataFrame containing exchange rates and other features.

    Returns:
        pd.DataFrame: Updated DataFrame with new features added for each pair.
    """
    output_df = update_targets(df, multipliers, widemultipliers, n=336)
    return output_df


storage_map = {'targets': ['target', 'hours_passed', 'buy_sl_time',
                           'sell_sl_time']}