import pandas as pd 
from src.feature_engineering import (
    FeatureEngineeringFactory,
    FrequencyEncoding,
    TargetEncoding,
    TimeSeriesFeatureEngineering,
    ConvertToFloat
)
from logging_config import logger 



def feature_engineering_step(data: pd.DataFrame)-> pd.DataFrame:
    """  
    Feature engineering step of the pipeline.
    """
    # params 
    freq_encoding_col = ['Flow ID'] # via analysis
    target_encoding_cols = ['Src IP', 'Dst IP'] # via analysis 
    time_col = ['Timestamp']

    # Frequency encoding 
    try:
        f_eng = FeatureEngineeringFactory(strategy=FrequencyEncoding(cat_cols=freq_encoding_col))
        logger.info("Feature Engineering initialized")
        data = f_eng.engineer_features(data)
        logger.info("--Frequency encoding applied")
    except Exception as e:
        print(f"Error: {e}")
        print("Frequency encoding step failed")
        logger.error("--Frequency encoding failed")
    # Target encoding 
    try:
        f_eng.set_strategy(strategy=TargetEncoding(features=target_encoding_cols, target=str(data.columns[-1])))
        data = f_eng.engineer_features(data)
        logger.info("--Target encoding applied")
    except Exception as e:
        print(f"Error: {e}")
        print("Target encoding step failed") 
        logger.error("--Target encoding failed")
    # Time series feature engineering
    try:
        f_eng.set_strategy(strategy=TimeSeriesFeatureEngineering(features=time_col,target=str(data.columns[-1])))
        data = f_eng.engineer_features(data)
        logger.info("--Time series feature engineering applied")
    except Exception as e:
        print(f"Error: {e}")
        print("Time series feature engineering step failed")
        logger.error("--Time series feature engineering failed")
    # Convert to float 
    try:
        f_eng.set_strategy(strategy=ConvertToFloat())   
        data = f_eng.engineer_features(data)
        logger.info("--Convert to float applied")
    except Exception as e:
        print(f"Error: {e}")
        print("Convert to float step failed")
        logger.error("--Convert to float failed")
        
    return data






