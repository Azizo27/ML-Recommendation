from EncodingFeatures import EncodingAllFeatures
import pandas as pd
import numpy as np
import os

# Define a global variable to store the transformed DataFrame
stored_dataToPredict = None

def TransformDfToPredict(DfToPredict, month):
    global stored_dataToPredict

    # Return the stored DataFrame if it exists
    if stored_dataToPredict is not None:
        return stored_dataToPredict.copy()

    # Perform the transformation on the first call
    dataToPredict = EncodingAllFeatures(DfToPredict)
    Features_filename = 'FittedFeaturesofModels.txt'
    with open(os.path.join(month, Features_filename), 'r') as file:
        train_features_name = file.read().split(',')

    column_names_df = dataToPredict.columns.tolist()

    for column in column_names_df:
        if column not in train_features_name:
            if column in dataToPredict:
                del dataToPredict[column]

    for column in train_features_name:
        if column not in column_names_df:
            dataToPredict[column] = 0

    dataToPredict = dataToPredict.reindex(columns=train_features_name)

    # Store the transformed DataFrame for future calls
    stored_dataToPredict = dataToPredict

    return dataToPredict.copy()
