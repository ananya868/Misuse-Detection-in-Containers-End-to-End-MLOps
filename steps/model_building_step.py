import pandas as pd 
from src.model_building import (
    GradientBoostingClassifierModel,
    KNNClassifierModel,
    LightGBMClassifierModel,
    RandomForestModel, 
    SVMModel, 
    XGBoostModel
)
from src.model_building import ModelBuildingFactory
from logging_config import logger 



def model_building_step(X_train, y_train, X_test, model_name: str):
    """
    Model building step
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train should have the same number of rows")
        logger.error("--X_train and y_train should have the same number of rows")

    logger.info("Model Building initiated")
    # Models
    if model_name == "Gradient Boosting Classifier": 
        try:
            model = ModelBuildingFactory(model=GradientBoostingClassifierModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--Gradient Boosting Classifier model built successfully")
        except Exception as e: 
            print(f"Error: {e}")
            print("--Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    elif model_name == "K Nearest Neighbors Classifier": 
        try:
            model = ModelBuildingFactory(model=KNNClassifierModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--K Nearest Neighbors Classifier model built successfully")
        except Exception as e: 
            print(f"Error: {e}")
            print("Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    elif model_name == "LightGBM Classifier":
        try:
            model = ModelBuildingFactory(model=LightGBMClassifierModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--LightGBM Classifier model built successfully")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    elif model_name == "Random Forest Classifier":
        try:
            model = ModelBuildingFactory(model=RandomForestModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--Random Forest Classifier model built successfully")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    elif model_name == "Support Vector Classifier":
        try:
            model = ModelBuildingFactory(model=SVCModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--Support Vector Classifier model built successfully")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    elif model_name == "XGBoost Classifier":
        try:
            model = ModelBuildingFactory(model=XGBoostModel())
            prediction, model_obj = model.train_model(X_train, y_train, X_test)
            logger.info("--XGBoost Classifier model built successfully")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to build model: {model_name}")
            logger.error("--Failed to build model: {model_name}")
    else:
        raise ValueError("Invalid model name")
        logger.error("--Invalid model name")


    return prediction, model_obj

