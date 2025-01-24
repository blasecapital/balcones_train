# test_data_preparation_create_targets.py

'''
Create targets test/template for development:
    
    Accept the pd.DataFrame from the iteration's base_data table for 
    target engineering.
    
    Map the targets to their storage location.
    
    Calculate each target and add each to a targets function to
    package for easy import into CreateTargets.
'''

import numpy as np


def required(func):
    func.required = True
    return func


def target1(df):
    condition = df['close'] > df['open']
    df['target'] = np.where(condition, 1, 0)
    return df
    

@required
def targets(df):
    target1(df)
    return df


'''required'''
storage_map = {'test_feature_data': ['target']}
