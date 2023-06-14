

from CreatingModelProduct import CreatingModelProduct
from LoadCsv import LoadCsv
from TransformDfToPredict import TransformDfToPredict
import pandas as pd
import numpy as np
import pickle
import os


def PredictProbabilityProduct(dfToPredict, features, target, model_file_name, FileToCreateModel):
    
    #If the model does not exist, create it
    if not os.path.exists(target):
        print("Creating Subfolders for the target "+target +" ...")
        os.makedirs(target, exist_ok=True)
        print("Model does not exist, creating it...")
        dfForTraining = LoadCsv(FileToCreateModel, FileToCreateModel)
        CreatingModelProduct(dfForTraining, model_file_name, features, target)
    
    print("Loading model "+model_file_name+ "...")

    dataToPredict = TransformDfToPredict(dfToPredict)
    
    with open(os.path.join(target, model_file_name), "rb") as file:
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
    
    dataToPredict["customer_code"] = dfToPredict.loc[dataToPredict.index, "customer_code"]
    
    df_selected = dataToPredict[['customer_code', target]]
    df_selected.to_csv(os.path.join(target, 'prediction_'+target+'.csv'), index=False)



