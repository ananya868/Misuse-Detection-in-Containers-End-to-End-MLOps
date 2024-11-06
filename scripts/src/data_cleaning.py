import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract class for Data cleaner 
class DataCleaner(ABC):
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to clean data."""
        pass



# Define a class for Data Cleaner
class DropEmptyFeatures(DataCleaner):
    """
    Drop columns with no unique values.
    """
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with all values same."""
        return data.loc[:, data.nunique() != 1]


# Define a class for Dropping Missing Values 
class DropMissingValues(DataCleaner):
    """  
    Drop rows with missing values.
    """
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values."""
        print(f"[info] --number of rows before dropping missing values: {data.shape[0]}--")
        return data.dropna()


# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(DataCleaner):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def clean(self, data: pd.DataFrame):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        df_cleaned = data.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            if fill_value is None:
                print("[warning] --No fill value provided. Using 'None' in place of constant--")
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        # logging.info("Missing values filled.")
        return df_cleaned



# Context Class for Handling Missing Values
class DataCleanerFactory:
    def __init__(self, strategy: DataCleaner):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataCleaner):
        """
        Changes the strategy dynamically.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        # logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.
        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        # logging.info("Executing missing value handling strategy.")
        return self._strategy.clean(data)