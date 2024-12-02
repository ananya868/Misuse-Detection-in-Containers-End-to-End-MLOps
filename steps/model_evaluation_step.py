import pandas as pd 
from src.model_evaluation import (
    ModelEvaluationFactory, 
    AccuracyScore, 
    ConfusionMatrix, 
    ClassificationReport
)



def model_evaluation_step(y_true, y_pred, metrics: str="accuracy"):
    """  
    Evaluate the model using the accuracy score, confusion matrix, and classification report
    """  
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("The number of rows in y_true and y_pred must be the same")
    
    # Create a model evaluation factory 
    if metrics == "accuracy": 
        try:
            evaluator = ModelEvaluationFactory(strategy=AccuracyScore(y_true, y_pred))
            score = evaluator.evaluate()
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get accuracy score")
    elif metrics == "confusion matrix":
        try:
            evaluator = ModelEvaluationFactory(strategy=ConfusionMatrix(y_true, y_pred))
            score = evaluator.evaluate()
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get confusion matrix")
    elif metrics == "classification report":
        try:
            evaluator = ModelEvaluationFactory(strategy=ClassificationReport(y_true, y_pred))
            score = evaluator.evaluate()
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get classification report")
    else:
        raise ValueError("Invalid metric. Please select 'accuracy', 'confusion matrix', or 'classification report'")

    return score
