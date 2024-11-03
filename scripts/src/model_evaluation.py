import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from abc import ABC, abstractmethod


# abstract class for Model Evaluation 
class ModelEvaluation(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass



# Define a class for Accuracy score
class AccuracyScore(ModelEvaluation):
    def __init__(self, y_true, y_pred):
        """  
        y_true: the true target values
        y_pred: the predicted target values
        """
        self.y_true = y_true
        self.y_pred = y_pred
    
    def evaluate(self):
        """
        Evaluate the model using the accuracy score 
        """
        return accuracy_score(self.y_true, self.y_pred)
    
    
# Define a class for confusion matrix 
class ConfusionMatrix(ModelEvaluation):
    def __init__(self, y_true, y_pred):
        """  
        y_true: the true target values
        y_pred: the predicted target values
        """
        self.y_true = y_true
        self.y_pred = y_pred
    
    def evaluate(self):
        """
        Evaluate the model using the confusion matrix 
        """
        return confusion_matrix(self.y_true, self.y_pred)


# Define a class for classification report
class ClassificationReport(ModelEvaluation):
    def __init__(self, y_true, y_pred):
        """  
        y_true: the true target values
        y_pred: the predicted target values
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate(self):
        """
        Evaluate the model using the classification report 
        """
        return classification_report(self.y_true, self.y_pred)



# Concrete Strategy for Model Evaluation 
class ModelEvaluationFactory:
    def __init__(self, strategy: ModelEvaluation):
        """  
        Initialize the model evaluation 
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelEvaluation):
        """
        Switch the model evaluation to the new strategy
        """
        self._strategy = strategy
    
    def evaluate(self):
        """
        Evaluate the model using the selected strategy
        """
        return self._strategy.evaluate()

        
    

        
    
