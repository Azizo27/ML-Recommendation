
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os

def predict_segmentation(df):
    model_file_name = 'segmentationModel.pkl'
    features = ['age']
    target= 'segmentation'
    
    if os.path.exists(model_file_name):
        print("Loading the saved model from the file...")
        model = pickle.load(open(model_file_name,'rb'))
    else:
        print('Preprocessing the data...')
        selected_columns = features + [target]
        data = df[df[target].notnull()][selected_columns].copy()
        
        print('Removing the target variable...')
        X = data.drop(columns=[target])
        y = data[target]
        
        print('Splitting the data into training and testing sets...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Training the random forest model...')
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        print('Making predictions on the test set...')
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        
        print("Saving the trained model to a file...")
        pickle.dump(model, open(model_file_name,'wb'))
        
    
    
    Null_segmentation = df[df[target].isnull()][features] # Get the row with missing values in segmentation
    df.loc[df[target].isnull(), target] = model.predict(Null_segmentation)
        
    return df
    
