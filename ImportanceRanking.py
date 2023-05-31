

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from DataCleaning import LoadCsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os


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


# Load dataset
def CreatingModelProduct(df, model_file_name, features, target):
    print("Starting...")
    
    X = EncodingAllFeatures(df[features])
    with open(os.path.join(target, 'FittedFeaturesof'+target+'.txt'),'w') as file:
        column_names = X.columns.tolist()
        file.write(','.join(str(item) for item in column_names))

            
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    print("Saving the trained model to a file...")
    with open(os.path.join(target, model_file_name), "wb") as file:
        pickle.dump(model, file)
    
def PredictProbabilityProduct(dfToPredict, features, target, model_file_name, FileToCreateModel):
    
    #If the model does not exist, create it
    if not os.path.exists(target):
        print("Creating Subfolders for the target "+target +" ...")
        os.makedirs(target, exist_ok=True)
        print("Model does not exist, creating it...")
        dfForTraining = LoadCsv(FileToCreateModel, FileToCreateModel)
        CreatingModelProduct(dfForTraining, model_file_name, features, target)
    
    print("Loading model "+model_file_name+ "...")

    dataToPredict = TransformDfToPredict(target, dfToPredict)
    
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



dfTopredict = LoadCsv("Cleaned_Renamed_test_ver2.csv", "Cleaned_Renamed_test_ver2.csv")

features = ['age', 'gross_income', 'customer_seniority', 'customer_relation_type_at_beginning_of_month', 'segmentation', 'gender']
FileToCreateModel = "Cleaned_Renamed_smalltrain_ver2.csv"

'''
all_products=  [ "product_savings_account", "product_guarantees", "product_current_accounts",
    "product_derivada_account", "product_payroll_account", "product_junior_account",
    "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
    "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
    "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
    "product_loans", "product_taxes", "product_credit_card", "product_securities",
    "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]
'''

all_products=  [ "product_current_accounts","product_payroll_account"]


for target in all_products:
    model_file_name = 'model_' + target + '.pkl'
    PredictProbabilityProduct(dfTopredict, features, target, model_file_name, FileToCreateModel)
    print("Done with", target)