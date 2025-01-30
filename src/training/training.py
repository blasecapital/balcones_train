# training.py


from components import CleanRawData, PrepData, Train


def training(clean=True, prep=False, train=False):
    """
    Orchestrate the entire process of loading and preprocessing the training data,
    defining model architectures and custom functions, and saving results
    and supporting files for model evaluation.
    
    LoadTrainingData:
        load_data returns a dictionary of all the feature DataFrames
    """
    if clean:
        # Initialize the preprocess object
        clean = CleanRawData()
        
        clean.inspect_data(
            data_keys=['iter0_targets'],
            describe_features=False,
            describe_targets=True,
            target_type='cat',
            plot_features=False,
            plot_mode='stat',
            plot_skip=12
            )
        
    if prep:
        prep = PrepData()
        prep.prep()
        
    if train:
        train = Train()
        
    
if __name__ == "__main__":
    training()
    