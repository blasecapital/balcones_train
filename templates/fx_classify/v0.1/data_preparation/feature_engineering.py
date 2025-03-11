# feature_engineering.py

'''
Create features test/template for development:
    
    Accept the pd.DataFrame from the iteration's raw data table for 
    feature engineering.
    
    Map the features to their storage location.
    
    Calculate each feature and add each to a features function to
    package for easy import into CreateFeatures.
'''

import pandas as pd
import numpy as np


#########################################################
# Individual feature calculation functions
#########################################################


def feature1():
    # Connect to raw database
    # Load data into df
    # Save transformation to a df column
    df = pd.DataFrame()
    df['feature1'] = np.zeros(3)
    return df['feature1']


def feature2():
    df = pd.DataFrame()
    df['feature2'] = np.ones(3)
    return df['feature2']


#########################################################
# Required functions for the data_preparation.py module
#########################################################


def features():
    '''
    Return a single df by calling all feature functions from above.
    Each feature needs to be in its own column.
    '''
    df = pd.DataFrame()
    df['feature1'] = feature1()
    df['feature2'] = feature2()
    return df

storage_map = {
    'feature_table_name': [                        
        'feature1'],
    
    'another_table_name': [
        'feature2']
    }