# training.py


from components import ProcessRawData, Train #LoadTrainingData, 


def training(clean=True):
    """
    Orchestrate the entire process of loading and preprocessing the training data,
    defining model architectures and custom functions, and saving results
    and supporting files for model evaluation.
    
    LoadTrainingData:
        load_data returns a dictionary of all the feature DataFrames
    """
    # Initialize the preprocess object
    prd = ProcessRawData()
        
    if clean:
        prd.clean_data(
            data_keys=['iter0_hourly_features'],
            describe_features=True,
            describe_targets=False,
            plot_features=False,
            clean=False
            )
        
    '''
    # Old process
    ltd = LoadTrainingData()
    raw_data = ltd.load_data()
    
    prd = ProcessRawData(raw_data)
    training_data = prd.process_raw_data(
        filter_indices=True,
        engineering=True,
        engineering_mode='target',
        encode_categories=False,
        prepare_features=True,
        prepare_targets=True
        )
    
    train = Train(training_data)
    train.train_models()
    '''
if __name__ == "__main__":
    training()
    