# test_data_preparation_create_features.py

'''
Create features test/template for development:
    
    Accept the pd.DataFrame from the iteration's base_data table for 
    feature engineering.
    
    Map the features to their storage location.
    
    Calculate each feature and add each to a features function to
    package for easy import into CreateFeatures.
'''

import numpy as np


def required(func):
    func.required = True
    return func


def feature1(df):
    df['feature1'] = np.where(
        (df['high'] - df['open']) / df['open'] > 0.01, 1, 0)
    return df


def feature2(df):
    df['feature2'] = np.where(
        (df['open'] - df['low']) / df['open'] > 0.01, 1, 0)
    return df


def feature3(df):
    df['feature3'] = np.where(
        (abs(df['close'] - df['open']) / df['open']) > 0.015, 1, 0)
    return df


@required
def features(df):
    df = feature1(df)
    df = feature2(df)
    df = feature3(df)
    return df


'''required'''
storage_map = {
    'test_feature_data': ['feature1', 'feature2', 'feature3']
    }