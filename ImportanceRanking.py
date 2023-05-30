

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


def TransformDfToPredict(TrainFeaturesFile, DfToPredict):
    dataToPredict = EncodingAllFeatures(DfToPredict)
    
    with open(TrainFeaturesFile, 'r') as file:
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


# USELESS FUNCTION: Get feature names from a fitted model (NOT WORKING --> Get ['0', '1', '2'...])
def get_fitted_feature_names(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    if isinstance(model, dict):
        feature_names = model['feature_names']
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Assuming all trees in the forest have the same feature importances
        importances = model.estimators_[0].feature_importances_
        feature_names = np.arange(len(importances)).astype(str)
    else:
        raise ValueError("Could not find feature names in the model.")

    return feature_names

    
# Load dataset
def CreatingModelProduct(df, model_file_name, features, target):
    print("Starting...")
    
    X = EncodingAllFeatures(df[features])
    with open('FittedFeatures.txt', 'w') as file:
        column_names = X.columns.tolist()
        file.write(','.join(str(item) for item in column_names))

            
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    
    
    '''
    # Get feature importances and sort them in descending order
    print("Importances of each features...")
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {importances[idx]}")
    '''

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    print("Saving the trained model to a file...")
    pickle.dump(model, open(model_file_name,'wb'))
    
    
    '''
    print("Predicting probability of test for being buyed...")
    # Use predict_proba to get the probability
    probabilities = model.predict_proba(X_test)
    
    # SHOULD ALWAYS BE > 1 BECAUSE IF NOT, IT MEANS THAT THE ACCURACY IS 100% AND THIS IS NOT POSSIBLE WITH A BIG DATASET 
    # (Because It would mean that in the entire dataset, there is not a single client who bought the product)
    if probabilities.shape[1] > 1:
        X_test[target] = probabilities[:, 1]
    else:
        # If there is only one column, assign (1 - probabilities[:, 0]) to 'probability_buyed'
        #With this operation, probability_buyed will be equal to 1 if the client bought the product and 0 if he didn't
        X_test[target] = 1 - probabilities[:, 0]
    
    
    #To get the customer_code
    X_test["customer_code"] = df.loc[X_test.index, "customer_code"]
    
    
    df_non_zero = X_test[X_test[target] > 0.02].copy()
    if not df_non_zero.empty:
        df_non_zero.loc[:, "prediction_target"] = model.predict(df_non_zero.drop(target, axis=1))
        df_non_zero.loc[:, "real_target"] = y_test.loc[df_non_zero.index]
    print(df_non_zero[["customer_code", target]])
    
    
    print(X_test[["customer_code", target]])
    '''

def PredictProbabilityProduct(dfToPredict, features, target, model_file_name, FileToCreateModel):
    
    #If the model does not exist, create it
    if not os.path.exists(model_file_name):
        print("Model does not exist, creating it...")
        dfForTraining = LoadCsv(FileToCreateModel, FileToCreateModel)
        CreatingModelProduct(dfForTraining, model_file_name, features, target)
    
    print("Loading model "+model_file_name+ "...")

    dataToPredict = TransformDfToPredict('FittedFeatures.txt', dfToPredict)
    print("After Transform", dataToPredict.head(10))
    model = pickle.load(open(model_file_name,'rb'))
    
    
    '''
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[::-1]
    features = dfToPredict.columns[top_feature_indices]
    print("features of model: ", features)
    '''
    
    
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
    df_selected.to_csv('prediction_'+target+'.csv', index=False)


dfTopredict = LoadCsv("Cleaned_Renamed_test_ver2.csv", "Cleaned_Renamed_test_ver2.csv")

features = ['age', 'gross_income', 'customer_seniority', 'customer_relation_type_at_beginning_of_month', 'segmentation', 'gender']

target="product_current_accounts"
model_file_name = 'model_' + target + '.pkl'
FileToCreateModel = "Cleaned_Renamed_smalltrain_ver2.csv"

PredictProbabilityProduct(dfTopredict, features, target, model_file_name, FileToCreateModel)

'''
all_products=  [col for col in df.columns if col.startswith('product_')]
for target in all_products:
    model_file_name = 'model_' + target + '.pkl'
    CreatingModelProduct(df,model_file_name, features, target)
    print("Done with", target)

'''