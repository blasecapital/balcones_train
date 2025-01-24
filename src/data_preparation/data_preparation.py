# data_preparation.py


from components import CreateFeatures, CreateTargets


def data_preparation(features=False, targets=True):
    """
    Orchestrate the entire process of loading the desired data, applying
    feature engineering, and storing the features for model training.
    
    Prerequisites:
        - load_data requires data be stored in a database
        
    Returns:
        None: Data is stored in a database
        
    Quick start:
        1) Update /config.env which stores database paths
        2) Update /projects/{project name}/data_preparation/config.py which stores
           iteration's arguements           
    """
    
    if features:
        # Load base feature data, create features, and store them
        cf = CreateFeatures()
        cf.calculate_and_store_features()
    
    if targets:
        # Load base target data, create targets, and store them
        #target_df = ld.load_data(mode='targets')
        #ct = CreateTargets(target_df)
        #target_df = ct.create_targets()
        #ct.store_targets(target_df)
        ct = CreateTargets()
        ct.calculate_and_store_targets()
    

if __name__ == "__main__":
    data_preparation()
    