# training.py


from components import CleanRawData, PrepData, Train


def training(clean=False, prep=False, train=True):
    """
    Orchestrate the entire process of loading and preprocessing the training data,
    defining model architectures and custom functions, and saving results
    and supporting files for model evaluation.
    
    Cleaning Process:
        1) Inspect the features and targets to inform filter functions
            a. Begin with the raw feature and target database(s)
        2) Write the filter functions according to inspection observations
        3) Run the clean.clean() module to collect bad primary keys
        4) Run clean.align() module to save filtered data to clean database
        5) Revise config's source_query's database reference to point to new,
           clean database
        6) Loop starting at step 1 if necessary
            a. For step 1, change the database reference to the new, clean 
               database to observe the previous filter process's impact
            b. Ensure filter functions are holistic each time, the process
               is designed to filter the entire, raw database
               
        Key args:
            
            - Source: /<main path>/projects/<iteration folder>/training/config.py
            - Keys used: source_query, primary_key, data_processing_modules_path,
              clean_functions, bad_keys_path, align
            - Configure inspect, filter_keys, align flags according to your needs
              each run
              
            Take care to update the config.py file every time you run the 
            cleaning modules.
    
    Prep Process:
        
    Train Process:
    """
    if clean:
        # Initialize the preprocess object
        clean = CleanRawData()
        
        # Which clean processes do you want to run
        inspect = False
        filter_keys = False
        align = True
                
        if inspect:        
            clean.inspect_data(
                data_keys=['iter0_hourly_features'],
                describe_features=True,
                describe_targets=False,
                target_type='cat',
                plot_features=False,
                plot_mode='rate',
                plot_skip=12
                )
            
        if filter_keys:
            clean.clean(data_keys=['iter0_hourly_features'])
            
        if align:
            clean.align()
        
    if prep:
        # Initialize the data preparation object
        prep = PrepData()
        
        # Which prep processes do you want to run
        engineer=False
        scale=False
        prep_and_save=True
        validate_data=True
        
        if engineer:
            prep.engineer(mode='target')
        if scale:
            prep.scale()
        if prep_and_save:
            prep.save_batches()
        if validate_data:
            prep.validate_record()
        
    if train:
        train = Train()
        train.train_models()
        
    
if __name__ == "__main__":
    training()
    