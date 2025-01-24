# training.py


from components import LoadTrainingData, ProcessRawData, Train


def training():
    """
    Orchestrate the entire process of loading and preprocessing the training data,
    defining model architectures and custom functions, and saving results
    and supporting files for model evaluation.
    
    LoadTrainingData:
        load_data returns a dictionary of all the feature DataFrames
    """
    ltd = LoadTrainingData()
    raw_data = ltd.load_data()
    
    prd = ProcessRawData(raw_data)
    training_data = prd.process_raw_data(
        filter_indices=True,
        engineering=True,
        engineering_mode='feature',
        encode_categories=False,
        prepare_features=True,
        prepare_targets=True
        )
    
    train = Train(training_data)
    train.train_models()
    
    
if __name__ == "__main__":
    training()
    