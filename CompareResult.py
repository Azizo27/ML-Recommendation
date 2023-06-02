import pandas as pd
from LoadCsv import LoadCsv
import os

def CompareWithPrediction(dfWithRealValue):
    all_subfolders = [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name)) and name.startswith("product_")]
    df_test = dfWithRealValue.copy()


    for target in all_subfolders:
        df_predicted = pd.read_csv(f"{target}/{'prediction_'+target+'.csv'}")
        
        df_predicted[target] = df_predicted[target].apply(lambda x: 1 if x > 0.5 else 0)
        
        buyed1 = df_predicted[target]
        buyed2 = df_test[target]

        similarity = (buyed1 == buyed2).mean() 
        percentage = similarity*100
        print("The similarity of the "+target+" column is: ",percentage, "%")


'''
df= LoadCsv("Cleaned_Renamed_train_May2016.csv", "Cleaned_Renamed_train_May2016.csv")
CompareWithPrediction(df)
'''