# Assuming you have a pandas DataFrame called 'data' containing your dataset
import pandas as pd
import numpy as np
from DataCleaning import LoadCsv
from IncomeCleaning import *
from Useless.IncomePrediction import predict_income


def select_featuresVER2(data, target, threshold):
    # Select numerical columns from the data
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Select non-numerical columns and convert them to one-hot encoding
    categorical_cols = list(set(data.columns) - set(numerical_cols))
    encoded_data = pd.get_dummies(data[categorical_cols])

    # Concatenate numerical and encoded categorical columns
    processed_data = pd.concat([data[numerical_cols], encoded_data], axis=1)

    # Calculate the correlation matrix for processed data
    corr_matrix = processed_data.corr()

    # Sort the correlations with the target variable in descending order
    correlations = abs(corr_matrix[target]).sort_values(ascending=False)

    # Select features with correlation above the threshold
    selected_features = correlations[correlations > threshold].index.tolist()

    return selected_features


def select_featuresVER2NUMERICALONLY(data, target, threshold):
    # Select numerical columns from the data
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate the correlation matrix for numerical features
    corr_matrix = data[numerical_cols].corr()

    # Sort the correlations with the target variable in descending order
    correlations = abs(corr_matrix[target]).sort_values(ascending=False)

    # Select features with correlation above the threshold
    selected_features = correlations[correlations > threshold].index.tolist()

    return selected_features


data = LoadCsv("Cleaned_Renamed_biggertrain_ver2_WithoutIncome.csv", "Cleaned_Renamed_biggertrain_ver2_WithoutIncome.csv")

data = replace_null_gross_incomeUsingStartDate(data)

print("Starting...")
# Specify the target variable name
target_variable = "gross_income"

# Set the threshold for feature selection
correlation_threshold = 0.3

'''
# Call the function to select the pertinent features
selected_features = select_featuresVER2(data, target_variable, correlation_threshold)

# Print the selected features
print("Selected Features:", selected_features)
'''

# Call the function to select the pertinent features
Numerical_selected_features = select_featuresVER2NUMERICALONLY(data, target_variable, correlation_threshold)

# Print the selected features
print("Selected Features:", Numerical_selected_features)
data = data.dropna(subset=['gross_income'])
predict_income(data)

