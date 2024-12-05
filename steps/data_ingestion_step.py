import pandas as pd 
from src.data_ingestion import DataIngestionFactory
from logging_config import logger 


def data_ingestion_step(file_path: str)-> pd.DataFrame:
    """
    Ingest data from a file and return a DataFrame.
    """
    try:
        factory = DataIngestionFactory() # Factory
        ingestor = factory.get_ingestor(file_path) # Retrieve the suitable ingestor
        data = ingestor.ingest(file_path) # get data 
        logger.info("Data Ingested")
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")

    return data


