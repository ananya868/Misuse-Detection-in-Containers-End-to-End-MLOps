import pandas as pd 
from src.data_cleaning import (
    DataCleanerFactory,
    DropEmptyFeatures,
    FillMissingValuesStrategy, 
    DropMissingValues
)
from logging_config import logger 


def data_cleaning_step(data: pd.DataFrame, method: str='drop', fill_value: int=None)-> pd.DataFrame:
    """
    Clean the data by dropping empty features and handling missing values.
    """
    # Drop empty features (Mandatory Step)
    try:
        cleaner = DataCleanerFactory(strategy=DropEmptyFeatures()) # Factory
        logger.info("Dropping empty features")
        data = cleaner.handle_missing_values(data) # Drop empty features
        logger.info("--Drop empty features step completed")
    except Exception as e:
        print(f"Error: {e}")
        print("Drop empty features step failed")
        logger.error("--Drop empty features step failed")

    # Missing values 
    if method == "drop": # preferred step
        try:
            cleaner.set_strategy(strategy=DropMissingValues())
            data = cleaner.handle_missing_values(data) # Drop all rows with missing values 
            logger.info("--Drop missing values step completed")
        except Exception as e:
            print(f"Error: {e}")
            print("Drop missing values step failed")
            logger.error("--Drop missing values step failed")
    elif method in ["mean", "median", "mode", "constant"]:
        try:
            cleaner.set_strategy(strategy=FillMissingValuesStrategy(method=method, fill_value=fill_value))
            data = cleaner.handle_missing_values(data) # Fill missing values with the specified method
            logger.info("--Fill missing values step completed")
        except Exception as e:
            print(f"Error: {e}")
            print("Fill missing values step failed")
            logger.error("--Fill missing values step failed")
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {method}")
        logger.error(f"Unsupported missing value handling strategy: {method}")

    return data