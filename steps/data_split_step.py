import pandas as pd 
from src.data_split import (
    RandomSplitterFactory,
    RandomSplitter
)



def data_split_step(data: pd.DataFrame, print_shapes: bool=True)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets."""    
    # The data-frame received should be cleaned, pre-processed, features engineered and features selected 
    # Also, the splitter assumes that the last column is the target column

    if data is None:
        raise ValueError("No data to split")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data is not a DataFrame")

    # mandatory steps 
    try: 
        splitter = RandomSplitterFactory(strategy=RandomSplitter())
        X_train, X_test, y_train, y_test = splitter.split(data)
    except Exception as e:
        print(f"Error: {e}")
        print("Data split step failed")
    
    # print shapes 
    if print_shapes:
        print(f"x_train: {X_train.shape} | x_test: {X_test.shape}")
        print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test
    

