import pandas as pd 
from src.feature_selection import (
    FeatureSelectionFactory,
    RobustScaling,
    LogTransformation,
    DimensionalityReduction, 
    UnderSampling,
    OverSampling
)
from logging_config import logger 



def feature_selection_step(data: pd.DataFrame, scaling_method: str='log')-> pd.DataFrame:
    """
    Feature selection step of the pipeline.
    """
    # params
    n_comps = 9 # Ideal PCA Components # Found via analysis done in the EDA section 
    oversampling_strategy = {8: 4000, 6: 3000, 3: 2000, 4: 2000, 7: 2000, 10: 1800, 5: 1500, 9: 1500} 
    # (pre decided via data analysis done in the EDA section)
    undersampling_strategy = {0: 10000, 1: 8000, 2: 5000} # this is decided via analysis on data 

    if data is None:
        raise ValueError("data is NoneType. Please provide a non-null pandas DataFrame.")
        logger.error("--data is NoneType. Please provide a non-null pandas DataFrame.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
        logger.error("--data must be a pandas DataFrame.")
    
    # Mandatory steps
    # Scaling | any one of the methods to be used
    if scaling_method == 'log':
        try:
            selector = FeatureSelectionFactory(strategy=LogTransformation())
            logger.info("Feature Selection initialized")
            data = selector.select_feature(data) 
            logger.info("--log transformation step completed")
        except Exception as e: 
            print(f"Error: {e}")
            print("Log Transformation step failed")
            logger.error("--Log Transformation step failed")
    elif scaling_method == "robust":
        try:
            selector = FeatureSelectionFactory(strategy=RobustScaling())
            data = selector.select_feature(data)
            logger.info("--Robust Scaling step completed")
        except Exception as e: 
            print(f"Error: {e}")
            print("Robust Scaling step failed")
            logger.error("--Robust Scaling step failed")

    # Dimensionality Reduction 
    try:
        selector.set_strategy(strategy=DimensionalityReduction(n_components=n_comps))
        data = selector.select_feature(data)
        logger.info("--Dimensionality Reduction step completed")
    except Exception as e:
        print(f"Error: {e}")
        print("Dimensionality Reduction step failed")
        logger.error("--Dimensionality Reduction step failed")
    
    # Sampling (sampling strategies are declared inside the src files) | mandatory steps
    try:
        selector.set_strategy(strategy=UnderSampling(sampling_strategy=undersampling_strategy))
        data = selector.select_feature(data)
        logger.info("--UnderSampling step completed")
    except Exception as e:
        print(f"Error: {e}")
        print("Under Sampling step failed. Might be due of the sampling strategy")
        logger.error("--Under Sampling step failed. Might be due of the sampling strategy")
    try:
        selector.set_strategy(strategy=OverSampling(sampling_strategy=oversampling_strategy))
        data = selector.select_feature(data)
        logger.info("--OverSampling step completed")
    except Exception as e:
        print(f"Error: {e}")
        print("Over Sampling step failed. Might be due to the sampling strategy")
        logger.error("--Over Sampling step failed. Might be due to the sampling strategy")

    return data


    

