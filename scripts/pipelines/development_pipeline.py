from steps.data_ingestion_step import data_ingestion_step
from steps.data_cleaning_step import data_cleaning_step
from steps.data_preprocessing_step import data_preprocessing_step 
from steps.feature_engineering_step import feature_engineering_step
from steps.feature_selection_step import feature_selection_step
from steps.data_split_step import data_split_step
from steps.model_building_step import model_building_step 
from steps.model_evaluation_step import model_evaluation_step



def development_pipeline(): 
    """  
    Complete development pipeline for ML model lifecycle.
    It orchestrates the steps of the pipeline.
    """
    # Step 1: Data Ingestion
    data = data_ingestion_step(file_path='dataset.csv')

    # Step 2: Data Cleaning
    data = data_cleaning_step(data)

    # Step 3: Data Preprocessings
    data = data_preprocessing_step(data)
    
    # Step 4: Feature Engineering
    data = feature_engineering_step(data)

    # Step 5: Feature Selection
    data = feature_selection_step(data)

    # Step 6: Data Splitting
    X_train, X_test, y_train, y_test = data_split_step(data)

    # Step 7: Model Building
    predictions, model = model_building_step(X_train, y_train)

    # Step 8: Model Evaluation
    score = model_evaluation_step(y_test, predictions)

    return model, score



if __name__ == '__main__':
    pass  # development_pipeline()