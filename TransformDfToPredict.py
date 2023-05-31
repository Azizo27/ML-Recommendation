from EncodingFeatures import EncodingAllFeatures
import pandas as pd
import numpy as np
import os

def TransformDfToPredict(target, DfToPredict):
    dataToPredict = EncodingAllFeatures(DfToPredict)
    
    with open(os.path.join(target, 'FittedFeaturesof'+target+'.txt'), 'r') as file:
        train_features_name = file.read().split(',')

    column_names_df = dataToPredict.columns.tolist()

    for column in column_names_df:
        if column not in train_features_name:
            del dataToPredict[column]
            
    for column in train_features_name:
        if column not in column_names_df:
            dataToPredict[column] = 0
    
    dataToPredict = dataToPredict.reindex(columns=train_features_name)
    print( 'the column of test after transforming: ', dataToPredict.columns.tolist())
    return dataToPredict
