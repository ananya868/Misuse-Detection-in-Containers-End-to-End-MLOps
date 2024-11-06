import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract class for Data Preprocessing 
class DataPreprocessing(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to preprocess data."""
        pass



# Define a class for Data Preprocessor
class RemoveOutliers(DataPreprocessing):
    def __init__(self, feature: str):
        """
        Initializes the RemoveOutliers with a specific feature.

        Parameters:
        feature (str): The feature to remove outliers from.
        """
        self._feature = feature

    def process(self, data: pd.DataFrame)-> pd.DataFrame:
        """Remove outliers from the data."""
        q75 = data[column].quantile(0.75)
    
        # Return DataFrame with values <= 75th percentile
        return data[data[self._feature <= q75]]


# Define a class for cap outliers 
class CapOutliers(DataPreprocessing):
    def __init__(self, features: list[str]):
        """
        Initializes the CapOutliers with a specific feature.

        feature (str): The feature to cap outliers from.
        """
        self._features = features
    
    def process(self, data: pd.DataFrame)-> pd.DataFrame:
        """Cap outliers from the data."""
        df_capped = data.copy()
    
        # Process each specified column
        for column in self._features:
            # Calculate 75th percentile
            q75 = data[column].quantile(0.75)
            
            # Cap values above 75th percentile
            df_capped[column] = data[column].clip(upper=q75)
        
        return df_capped 


# Define a class for capping with a value 
class CapWithValues(DataPreprocessing):
    def __init__(self, cap_dict: dict):
        """
        Initializes the CapWithValues with a specific feature.

        Parameters:
        cap_dict (dict): The dictionary with feature and cap value.
        """
        self._cap_dict = cap_dict
    
    def process(self, data: pd.DataFrame)-> pd.DataFrame:
        """Cap outliers from the data."""
        df_capped = data.copy()
    
        # Process each column with its specified cap value
        for column, cap_value in self._cap_dict.items():
            df_capped[column] = data[column].clip(upper=cap_value)
        
        return df_capped



# Concrete strategy for Data Preprocessing 
class DataPreprocessingFactory:
    def __init__(self, strategy: DataPreprocessing):
        """
        Initializes the DataPreprocessingFactory with a specific strategy.

        Parameters:
        strategy (DataPreprocessing): The strategy to be used for preprocessing data.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataPreprocessing):
        """
        Changes the strategy dynamically.

        Parameters:
        strategy (DataPreprocessing): The new strategy to be used for preprocessing data.
        """
        # logging.info("Switching data preprocessing strategy.")
        self._strategy = strategy

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processes the data using the current strategy.

        Parameters:
        data (pd.DataFrame): The input data to be preprocessed.

        Returns:
        pd.DataFrame: The preprocessed data.
        """
        return self._strategy.process(data)




