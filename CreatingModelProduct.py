
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
import os
from SplitWrapper import splitting_dataset

# Load dataset
def CreatingModelProduct(df, model_file_name, features, target):
    print("Starting...")

    X_train, X_test, y_train, y_test = splitting_dataset(df, features, target, 0.2, 42)

    print("Fitting...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    print("Saving the trained model to a file...")
    with open(os.path.join(target, model_file_name), "wb") as file:
        pickle.dump(model, file)
  