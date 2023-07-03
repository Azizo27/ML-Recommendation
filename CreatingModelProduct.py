
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
import os
from SplitWrapper import splitting_dataset, reset_first_time
from LoadCsv import LoadPerMonth
import gzip

# Load dataset
def CreatingModelProduct(df, features, target, month):
    print("Starting...")

    X_train, X_test, y_train, y_test = splitting_dataset(df, features, target, 0.2, 42, month)

    print("Fitting...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    with open(os.path.join(month, 'Accuracy.txt'), 'a') as file:
        file.write('\n'+ target + ' accuracy: ' + str(accuracy) + '\n')
    
    print("Saving the trained model to a compressed file...")
    with gzip.open(os.path.join(month, target + '.pkl.gz'), "wb") as file:
        pickle.dump(model, file)
  
  

# This function creates a subfolders for each montb. Then, it creates all 24 models in each of these subfolders.
# This function is 'skipped' if the subfolders already exist.
def CreatingAllMonthModels(all_products, all_months, features, train_file_name):
    
    for month in all_months:
        if not os.path.exists(month):
            print("Creating Subfolders for the month "+month +" ...")
            os.makedirs(month, exist_ok=True)
            
            df = LoadPerMonth(train_file_name, month) 
            for target in all_products:
                CreatingModelProduct(df, features, target, month)
            
            print("Resetting the first time...")
            reset_first_time()
            
        
        else:
            print("the folder "+month+" already exists")
    