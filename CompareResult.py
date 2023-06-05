import pandas as pd
from LoadCsv import LoadCsv
import os


def CountingZerosAndOne(df_test, df_predicted, target):
    print("\n\n")
    zero_count = df_test[target].value_counts()[0]
    pred_zero_count = df_predicted[target].value_counts()[0]
    commun_zero_values = ((df_predicted[target] == 0) & (df_test[target] == 0)).sum()
    
    one_count = df_test[target].value_counts()[1]
    pred_one_count = df_predicted[target].value_counts()[1]
    commun_one_values = ((df_predicted[target] == 1) & (df_test[target] == 1)).sum()

    print("Number of zeros for "+target + " in the REAL CASE :", zero_count)
    print("Number of zeros for "+target + " in the PREDICTION CASE :", pred_zero_count)
    print("Number of commun zero values for "+target+" :", commun_zero_values)
    
    print("\nNumber of ones for "+target + " in the REAL CASE :", one_count)
    print("Number of ones for "+target + " in the PREDICTION CASE :",pred_one_count)
    print("Number of commun one values for "+target+" :", commun_one_values)
    
def CompareWithPrediction(dfWithRealValue):
    all_subfolders = [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name)) and name.startswith("product_")]
    df_test = dfWithRealValue.copy()


    for target in all_subfolders:
        df_predicted = pd.read_csv(f"{target}/{'prediction_'+target+'.csv'}")
        
        df_predicted[target] = df_predicted[target].apply(lambda x: 1 if x > 0.5 else 0)
        
        CountingZerosAndOne(df_test, df_predicted, target)
        
        buyed1 = df_predicted[target]
        buyed2 = df_test[target]

        similarity = (buyed1 == buyed2).mean() 
        percentage = similarity*100
        print("The similarity of the "+target+" column is: ",percentage, "%")


df= LoadCsv("Cleaned_Renamed_train_May2016.csv", "Cleaned_Renamed_train_May2016.csv")
CompareWithPrediction(df)