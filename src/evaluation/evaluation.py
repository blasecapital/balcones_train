# evaluation.py


from components import Eval


def evaluation():
    """
    
    """
    e = Eval()
    
    pred=False
    explain=False
    metrics=False
    cal=False
    candidates=True
    
    if pred:
        e.predict_and_store(mode="test")
        
    if explain:
        e.explain()
    
    if metrics:
        e.report_metrics(display_mode='convert')
        
    if cal:    
        e.report_calibration(mode='convert')
        
    if candidates:
        e.report_candidates(
            mode='convert',
            accuracy_threshold=.35,
            volume=300)
    
if __name__ == "__main__":
    evaluation()
    