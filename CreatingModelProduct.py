from EncodingFeatures import EncodingAllFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
import os

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
  