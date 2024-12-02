import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd
import category_encoders as ce



# Define an abstract class for Feature Engineering
class FeatureEngineering(ABC):
    @abstractmethod
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to engineer features."""
        pass



# Define a class for Frequency encoding 
class FrequencyEncoding(FeatureEngineering):
    """  
    Frequency encoding of categorical columns.
    """
    def __init__(self, cat_cols: list):
        """
        Initializes the FrequencyEncoding with specific categorical columns.

        Parameters:
        cat_cols (list): The list of categorical columns to encode.
        """
        self.cat_cols = cat_cols
        
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Frequency encoding of categorical columns."""
        df_encoded = data.copy()
        
        # Process each column
        for column in self.cat_cols:
            # Calculate frequency of each category
            freq = df_encoded[column].value_counts(normalize=True)
            
            # Map frequency to each category
            df_encoded[column] = df_encoded[column].map(freq)
        
        return df_encoded


# Define a class for Target encoding
class TargetEncoding(FeatureEngineering):
    def __init__(self, features: list, target: str, smoothing=1.0):
        """
        Initializes the TargetEncoding with a specific target feature.

        Parameters:
        target (str): The target feature to encode.
        smoothing (float): The smoothing parameter for target encoding.    
        """
        self._features = features
        self._target = target
        self._smoothing = smoothing
    
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Target encoding of categorical columns."""
        df_encoded = data.copy()
        
        # Process each column
        for column in self._features:
            # Initialize the target encoder
            encoder = ce.TargetEncoder(cols=[column], smoothing=self._smoothing)
            
            # Fit and transform the column based on the target
            df_encoded[column] = encoder.fit_transform(df_encoded[column], df_encoded[self._target])

        return df_encoded
    

# Define a class for Engineering Time series feature
class TimeSeriesFeatureEngineering(FeatureEngineering):
    def __init__(self, features: list, target: str, format: str='ISO8601'):
        """
        Initializes the TimeSeriesFeatureEngineering with a specific feature.

        Parameters:
        feature (str): The feature to engineer.
        """
        self._features = features
        self._format = format
        self._target = target
        
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer time series features."""
        df_engineered = data.copy()

        # Process each specified column
        for column in self._features:
            # Convert to datetime
            df_engineered[column] = pd.to_datetime(df_engineered[column], format=self._format)
            
            # Extract time series features
            df_engineered["year"] = df_engineered[column].dt.year
            df_engineered["month"] = df_engineered[column].dt.month
            df_engineered["day"] = df_engineered[column].dt.day
            df_engineered["hour"] = df_engineered[column].dt.hour
            df_engineered["minute"] = df_engineered[column].dt.minute
            df_engineered["second"] = df_engineered[column].dt.second

            # Drop original Time series column
            if column in df_engineered.columns: 
                df_engineered.drop(column, axis=1, inplace=True)

            # Rearrange target label position to the end
            if self._target in df_engineered.columns and df_engineered.columns[-1] != self._target:
                cols = df_engineered.columns.tolist()
                # Remove 'Label' from its current position
                cols.remove('Label')
                # Append 'Label' to the end
                cols.append('Label')
                # Reindex the DataFrame with the new order of columns
                df_engineered = df_engineered[cols]

        return df_engineered


# Define a class for converting all int to float 
class ConvertToFloat(FeatureEngineering):
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert all integer columns to float."""
        df_converted = data.copy()
        
        # Identify columns with int32 and int64 data types
        int_cols = df_converted.select_dtypes(include=['int32', 'int64']).columns
        
        # Convert identified columns to float64
        df_converted[int_cols] = df_converted[int_cols].astype('float64')
        
        return df_converted



# Concrete strategy for Feature Engineering 
class FeatureEngineeringFactory:
    def __init__(self, strategy: FeatureEngineering):
        """  
        Initializes the FeatureEngineeringFactory with a specific strategy.

        Parameters:
            strategy (FeatureEngineering): The strategy to use for feature engineering
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineering):
        """
        Set the strategy for feature engineering.
    
        Parameters:
            strategy (FeatureEngineering): The new strategy to be used for preprocessing data.
        """
        # logging.info("Switching to different feature engineering strategy.")
        self._strategy = strategy

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering in the data using the current strategy.

        Parameters:
            data (pd.DataFrame): The input data to be processed.

        Returns:
            pd.DataFrame: Feature Engineered data.
        """
        return self._strategy.engineer(data)
        

