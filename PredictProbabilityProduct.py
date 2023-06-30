

from CreatingModelProduct import CreatingModelProduct
from LoadCsv import LoadCsv
from TransformDfToPredict import TransformDfToPredict
import pandas as pd
import numpy as np
import pickle
import os
import gzip


def PredictProbabilityProduct(dfToPredict, target, month):
    
    print("Loading model of "+target+ "...")

    dataToPredict = TransformDfToPredict(dfToPredict, month)
    
    print("Loading the trained model from the compressed file...")
    with gzip.open(os.path.join(month, target + '.pkl.gz'), "rb") as file:
        model = pickle.load(file)

    
    print("Predicting probability of test for being buyed...")
    # Use predict_proba to get the probability
    probabilities = model.predict_proba(dataToPredict)
    print("After Predict Probability")
    
    
    # SHOULD ALWAYS BE > 1 BECAUSE IF NOT, IT MEANS THAT THE ACCURACY IS 100% AND THIS IS NOT POSSIBLE WITH A BIG DATASET 
    # (Because It would mean that in the entire dataset, there is not a single client who bought the product)
    if probabilities.shape[1] > 1:
        dataToPredict[target] = probabilities[:, 1]
    else:
        # If there is only one column, assign (1 - probabilities[:, 0]) to 'probability_buyed'
        #With this operation, probability_buyed will be equal to 1 if the client bought the product and 0 if he didn't
        dataToPredict[target] = 1 - probabilities[:, 0]
    
    
    return dataToPredict[target]



