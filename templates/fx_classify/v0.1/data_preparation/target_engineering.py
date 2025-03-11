# target_engineering.py


'''
Create targets test/template for development:
    
    Accept the pd.DataFrame from the iteration's raw data table for 
    target engineering.
    
    Map the targets to their storage location.
    
    Calculate each target and add each to a targets function to
    package for easy import into CreateTargets.
'''


import pandas as pd
import numpy as np


#########################################################
# Individual target calculation functions
#########################################################


def calc_target():
    df = pd.DataFrame()
    df['target'] = np.random.rand(3)
    df['hours_passed'] = np.random.randint(3)
    return df

#########################################################
# Required functions for the data_preparation.py module
#########################################################


def targets():
    """
    Return a single df by calling all target functions from above.
    """
    output_df = calc_target()
    return output_df


storage_map = {
    # key: standard table name is 'targets'
    # itmes: column names in df
    'targets': ['target', 'hours_passed']}