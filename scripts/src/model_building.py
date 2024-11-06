import pandas as pd
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time 


# abstract class for Model Building 
class ModelBuilding(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self):
        pass



# Define a class for Gradient Boosting Classifier
class GradientBoostingClassifierModel(ModelBuilding):
    def __init__(self, n_estimators: int=100): 
        self.n_estimators = n_estimators
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators, # The number of boosting stages to be run
            learning_rate=0.1,  # Learning rate shrinks the contribution of each tree
            max_depth=3,  # Maximum depth of the individual regression estimators
            random_state=42 # Controls the random seed given at each base_estimator at each boosting iteration
            )   

    def train(self, X_train, y_train, X_test): 
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")

        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model


# Define a class for K Nearest Neighbors Classifier 
class KNNClassifierModel(ModelBuilding):
    def __init__(self, n_neighbors: int=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,  # Number of neighbors to consider for the vote
            weights='uniform',  # Weight function used in prediction
            algorithm='auto',  # Algorithm used to compute the nearest neighbors
            metric='minkowski',  # Distance metric to use
            p=1  # Power parameter for the Minkowski metric
        )
    
    def train(self, X_train, y_train, X_test):
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")

        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model


# Define a class for Light GBM classifier
class LightGBMClassifierModel(ModelBuilding):
    def __init__(self, n_estimators: int=300):
        self.n_estimators = n_estimators
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,  # Number of boosting rounds
            learning_rate=0.1,  # Boosting learning rate
            max_depth=1,  # Maximum tree depth for base learners
            random_state=42  # Random number seed
        )

    def train(self, X_train, y_train, X_test):
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")
        
        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model


# Define a class for RandomForest 
class RandomForestModel(ModelBuilding):
    def __init__(self, n_estimators: int=300):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,  # Number of trees in the forest
            criterion='gini',      # Use 'entropy' for information gain
            max_depth=None,        # Maximum depth of the tree
            min_samples_split=2,   # Minimum number of samples required to split an internal node
            min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node
            random_state=42,       # Random state for reproducibility
            n_jobs=-1  # Random number seed
        )
    
    def train(self, X_train, y_train, X_test):
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")
        
        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model


# Define a class for Support Vector Machine Model 
class SVMModel(ModelBuilding):
    def __init__(self, C: float=1.0, kernel: str='poly', gamma: str='auto'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = SVC(
            C=self.C,  # Regularization parameter
            kernel=self.kernel,  # Specifies the kernel type to be used in the algorithm
            gamma=gamma,  # Kernel coefficient
        ) 
    
    def train(self, X_train, y_train, X_test):
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")
        
        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model


# Define a class for XGBoost Classifier
class XGBoostModel(ModelBuilding):
    def __init__(self, n_estimators: int=800):
        self.n_estimators = n_estimators
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,  # Number of boosting rounds
            learning_rate=0.1,     # Step size shrinkage used to prevent overfitting
            max_depth=3,           # Maximum tree depth for base learners
            random_state=42,       # Random state for reproducibility
            use_label_encoder=False,  # Disable use of label encoder
            eval_metric='logloss'  # Evaluation metric
        )
    
    def train(self, X_train, y_train, X_test):
        """
        Train the model using the training data  
        """
        start = time.time()
        # fit model
        self.model.fit(X_train, y_train)
        end = time.time()
        print(f"[info] --Training time: {end - start} seconds--")
        
        # Make predictions 
        prediction = self.model.predict(X_test)
        return prediction, self.model



# Concrete strategy for Model Building 
class ModelBuildingFactory:
    def __init__(self, model: ModelBuilding):
        """  
        Initialize the model 
        """
        self._model = model

    def set_model(self, model: ModelBuilding):
        """
        Switch the model to the new model
        """
        self._model = model

    def train_model(self, X_train, y_train, X_test):
        """
        Train the model using the training data
        """
        return self._model.train(X_train, y_train, X_test)
