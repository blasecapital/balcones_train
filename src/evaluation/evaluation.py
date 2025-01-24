# evaluation.py


from components import LoadEvalData, ProcessEvalData, EvalInference


def evaluation():
    """
    Design so it may be used for inference on the original training data set
    and any new data set (forward testing data, for example).
    
    Requirements:
        - original load data config
        - original preprocessing config
        - original process_raw_data flags
        - supporting files (scalers, initial_bias, weights, etc.)
        - original create model function
        - evaluation config (export, which slices to inference, custom modules like decoding predictions)
    """
    # Load data
    led = LoadEvalData('model1')
    raw_data = led.load_data()
    
    # Preprocess data
    ped = ProcessEvalData(raw_data, 'model1')
    training_data = ped.process_raw_data()
        
    # Inference and save
    inf = EvalInference(training_data, 'model1')
    inf.eval_inference()
    
    # Calculate and report metrics
    
    
if __name__ == "__main__":
    evaluation()
    