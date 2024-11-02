import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract class for Data Preprocessing 
class DataPreprocessing(ABC):
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to preprocess data."""
        pass



# Define a class for Data Preprocessor
# --to be implemented in the next code cell--

