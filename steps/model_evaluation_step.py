import pandas as pd 
from src.model_evaluation import (
    ModelEvaluationFactory, 
    AccuracyScore, 
    ConfusionMatrix, 
    ClassificationReport
)
from logging_config import logger 



def model_evaluation_step(y_true, y_pred, metrics: str="accuracy"):
    """  
    Evaluate the model using the accuracy score, confusion matrix, and classification report
    """  
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("The number of rows in y_true and y_pred must be the same")
        logger.error("--The number of rows in y_true and y_pred must be the same")
    
    logger.info("Model Evaluation initialized")
    # Create a model evaluation factory 
    if metrics == "accuracy": 
        try:
            evaluator = ModelEvaluationFactory(strategy=AccuracyScore(y_true, y_pred))
            score = evaluator.evaluate()
            logger.info(f"Accuracy Score: {score}")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get accuracy score")
            logger.error("--Failed to get accuracy score")
    elif metrics == "confusion matrix":
        try:
            evaluator = ModelEvaluationFactory(strategy=ConfusionMatrix(y_true, y_pred))
            score = evaluator.evaluate()
            logger.info(f"Confusion Matrix: {score}")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get confusion matrix")
            logger.error("--Failed to get confusion matrix")
    elif metrics == "classification report":
        try:
            evaluator = ModelEvaluationFactory(strategy=ClassificationReport(y_true, y_pred))
            score = evaluator.evaluate()
            logger.info(f"Classification Report: {score}")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get classification report")
            logger.error("--Failed to get classification report")
    else:
        raise ValueError("Invalid metric. Please select 'accuracy', 'confusion matrix', or 'classification report'")
        logger.error("--Invalid metric. Please select 'accuracy', 'confusion matrix', or 'classification report'")

    return score
