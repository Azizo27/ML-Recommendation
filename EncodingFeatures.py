
import pandas as pd
import numpy as np


def EncodingAllFeatures(dataWithJustFeatures):
    
    # Split data into numeric features and target
    print("Splitting Numerical and Categorical Features...")
    numeric_cols = dataWithJustFeatures.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = dataWithJustFeatures[numeric_cols]

    # Convert non-numeric features to numeric using one-hot encoding
    print("Converting...")
    non_numeric_cols = dataWithJustFeatures.select_dtypes(exclude=[np.number]).columns.tolist()
    X_non_numeric = pd.get_dummies(dataWithJustFeatures[non_numeric_cols])

    # Concatenate numeric and non-numeric features
    print("Concatenating And Splitting...")
    X = pd.concat([X_numeric, X_non_numeric], axis=1)
    return X
