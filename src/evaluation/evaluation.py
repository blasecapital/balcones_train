# evaluation.py


from components import LoadEvalData, ProcessEvalData, Eval


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
    e = Eval()
    e.backtest_results()
    
if __name__ == "__main__":
    evaluation()
    