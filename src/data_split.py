import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split


# Define an abstract class for Data Splitting
class DataSplitter(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to split data."""
        pass



# Define a class for Random Splitting
class RandomSplitter(DataSplitter):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the RandomSplitter with a specific test size and random state.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self._test_size = test_size
        self._random_state = random_state 
    
    def split(self, data: pd.DataFrame) -> pd.DataFrame:
        """Randomly split the data into training and testing sets."""
        
        X = data.iloc[:, :-1]  # last column is the target
        y = data.iloc[:, -1] # last column is the target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=self._random_state)
        return X_train, X_test, y_train, y_test



# Concrete strategy for Random Splitter 
class RandomSplitterFactory: 
    def __init__(self, strategy: RandomSplitter):
        """  
        Initializes the RandomSplitterFactory with a specific strategy.
        """ 
        self._strategy = strategy
    
    def split(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Split the data using the strategy.
        """
        return self._strategy.split(data)



        

