import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Define a class for CSV Data Ingestor
class CsvDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a CSV file."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error: {e}")


# Define a class for ZIP Data Ingestor
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a ZIP file. Works for 1 file.
        # extracts the data at the same location 
        # as the ZIP file and reads the CSV file.
        """
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(file_path))
                csv_file = zip_ref.namelist()[0]
            return pd.read_csv(csv_file)
        except Exception as e: 
            print(f"Error: {e}")



# Define a class for Data Ingestion Factory
class DataIngestionFactory:
    @staticmethod
    def get_ingestor(file_path: str) -> DataIngestor:
        """Factory method to get the appropriate Data Ingestor."""
        if file_path.endswith('.zip'):
            return ZipDataIngestor()
        elif file_path.endswith('.csv'):
            return CsvDataIngestor()
        else:
            raise ValueError("Unsupported file format")


