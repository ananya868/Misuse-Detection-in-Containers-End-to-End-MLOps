import pandas as pd 
from src.data_preprocessing import (
    DataPreprocessingFactory,
    RemoveOutliers, 
    CapOutliers,
    CapWithValues
)
from logging_config import logger 



def data_preprocessing_step(data: pd.DataFrame)-> pd.DataFrame:
    """
    Pre-processes the data by capping outliers and replacing them with specified values.
    """
    # params 
    cap_cols = ['Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Flow Bytes/s', 'Flow IAT Mean', 'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance'] # found out via analysis
    cap_dict = {
        'Flow Packets/s': 200000,
        'Down/Up Ratio': 4,
        'Average Packet Size': 2500,
        'Fwd Segment Size Avg': 2500,
        'Bwd Segment Size Avg': 3000,
        'Fwd Bytes/Bulk Avg': 250000,
        'Subflow Fwd Bytes': 1700, 
        'Subflow Bwd Bytes': 2500,
        'Bwd Init Win Bytes': 12000,
        'Active Mean': 10000000,
        'Idle Mean': 40000000,
        'Total TCP Flow Time': 10000000000
    } # Decided via data analysis done in the EDA section 

    if data is None:
        # logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")
        logger.error("Received a NoneType DataFrame.")

    if not isinstance(data, pd.DataFrame):
        # logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")
        logger.error(f"Expected pandas DataFrame, got {type(data)} instead.")

    for i in cap_cols:
        if i not in data.columns: 
            # logging.error(f"Column '{i}' does not exist in the DataFrame.")
            raise ValueError(f"Column '{i}' does not exist in the DataFrame.")
            logger.error(f"Column '{i}' does not exist in the DataFrame.")
    
    # Mandatory steps 
    # cap outliers 
    try:
        processor = DataPreprocessingFactory(strategy=CapOutliers(features=cap_cols))
        logger.info("Pre Processing initialized")
        data = processor.preprocess_features(data)
        logger.info("--Processed data by capping outliers.")
    except Exception as e:
        print(f"Error: {e}")
        print("Cap Outliers step failed")
        logger.error("--Cap Outliers step failed")
    # cap outliers to specified value 
    try:
        processor.set_strategy(CapWithValues(cap_dict=cap_dict))
        data = processor.preprocess_features(data)
        logger.info("--Processed data by capping outliers with specified values.")
    except Exception as e:
        print(f"Error: {e}")
        print("Cap Outliers with Specified Value Failed")
        logger.error("--Cap Outliers with Specified Value Failed")
    
    return data

