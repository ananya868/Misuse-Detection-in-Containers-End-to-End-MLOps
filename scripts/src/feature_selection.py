from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter


# abstract class for Feature Selection
class FeatureSelection(ABC):
    @abstractmethod
    def select(self):
        """Abstract method to select features."""
        pass
    


# Define a class for robust scaling 
class RobustScaling(FeatureSelection):
    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scaling of numerical columns."""
        df_scaled = data.copy()
        
        # Separate the label and features before transformation
        y = df_scaled.iloc[:, -1]
        x = df_scaled.drop(columns='Label') # change target column if required
        # Process each column
        scaler = RobustScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

        # Add label column back to features 
        if 'Label' not in scaled_df.columns: 
            scaled_df['Label'] = y

        return scaled_df


# Define a class for Log transformation
class LogTransformation(FeatureSelection):
    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        """Log transformation of numerical columns."""
        df_scaled = data.copy()

        # Separate the label and features before transformation
        y = df_scaled.iloc[:, -1]
        features = df_scaled.drop(columns='Label') # change target column if required
        
        # Process each column
        features = features.applymap(lambda x: np.log1p(x) if x >= 0 else 0)

        if 'Label' not in features.columns:
            scaled_df = pd.concat([features, y], axis=1)
    
        return scaled_df


# Define a class for Dimensionality reduction using PCA 
class DimensionalityReduction(FeatureSelection):
    def __init__(self, n_components: int):
        # Note: The number of components is already determined via analysis done in the EDA section 
        self.n_components = n_components
    
    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dimensionality reduction using PCA."""
        # Separate the features and the target variable 
        X = data.iloc[:, :-1]  # last column is the target
        y = data.iloc[:, -1]

        # Process each column
        pca = PCA(n_components=self.n_components)
        X_reduced = pca.fit_transform(X)

        # Construct data frame
        column_names = [f'PC{i+1}' for i in range(self.n_components)]
        data_pca = pd.DataFrame(X_reduced, columns=column_names)    

        # Add label column back to the data
        if 'Label' not in data_pca.columns:
            data_pca['Label'] = y.values
        
        return data_pca


# Define a class for UnderSampling
class UnderSampling(FeatureSelection):
    def __init__(self, sampling_strategy: dict):
        self.sampling_strategy = sampling_strategy

    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        """UnderSampling of the majority class."""
        df_sampled = data.copy()

        X = df_sampled.drop(columns=['Label'])
        y = df_sampled['Label']

        # Apply Random UnderSampling to handle class imbalance
        rus = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        # Convert the resampled arrays back to a DataFrame
        data_undersampled = pd.DataFrame(X_res, columns=X.columns)
        data_undersampled['Label'] = y_res

        return data_undersampled


# Define a class for Over Sampling
class OverSampling(FeatureSelection):
    def __init__(self, sampling_strategy: dict): 
        self.sampling_strategy = sampling_strategy

    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        """OverSampling of the minority class."""
        df_sampled = data.copy()
        
        X = df_sampled.drop(columns=['Label'])
        y = df_sampled['Label']

        # Process each column
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Convert the resampled arrays back to a DataFrame
        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled['Label'] = y_res

        return df_resampled



# Concrete strategy for Feature Selection
class FeatureSelectionFactory:
    def __init__(self, strategy: FeatureSelection):
        """  
        Initialize the FeatureSelectionFactory with a strategy.
        """
        self._strategy = strategy
    

    def set_strategy(self, strategy: FeatureSelection):
        """
        Set the strategy for the FeatureSelectionFactory.
        """
        self._strategy = strategy

    def select_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Selection in the data using the current strategy.
        """
        return self._strategy.select(data)
