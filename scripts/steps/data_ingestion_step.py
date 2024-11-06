import pandas as pd 
from src.data_ingestion import DataIngestionFactory


def data_ingestion_step(file_path: str)-> pd.DataFrame:
    """
    Ingest data from a file and return a DataFrame.
    """
    factory = DataIngestionFactory(file_path) # Factory
    ingestor = factory.get_ingestor() # Retrieve the suitable ingestor
    data = ingestor.ingest(file_path) # get data 

    return data


