import pandas as pd
from LoadCsv import LoadCsv
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from TransformDfToPredict import TransformDfToPredict
from EncodingFeatures import EncodingAllFeatures

def DrawSimilarityBar(all_target, all_percentage):
    cmap = mcolors.LinearSegmentedColormap.from_list('CustomMap', ['blue', 'red'])
    
    fig, ax = plt.subplots()
    bars = ax.bar(all_target, all_percentage, color=cmap(all_percentage))
    
    plt.ylabel('Percentage')
    plt.title('Similarity between real and predicted cases for each product')
    plt.xticks(rotation=90, ha='right')
    
    # Add the y-axis values at the top of each bar
    for bar in bars:
        yval = bar.get_height()
        yval_lower = math.floor(yval * 100) / 100  # Round down to two decimal places
        ax.text(bar.get_x() + bar.get_width() / 2, yval, '{:.2f}'.format(yval_lower),
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=6)
    
    plt.tight_layout()
    plt.show()


    
    
def CountingZerosAndOne(df_test, df_predicted, target):
    print("\n\n")
    zero_count = df_test[target].value_counts()[0]
    pred_zero_count = df_predicted[target].value_counts()[0]
    commun_zero_values = ((df_predicted[target] == 0) & (df_test[target] == 0)).sum()
    
    try:
        one_count = df_test[target].value_counts()[1]
    except KeyError:
        one_count = 0
        
    try:
        pred_one_count = df_predicted[target].value_counts()[1]
    except KeyError:
        pred_one_count = 0
        
    try: 
        commun_one_values = ((df_predicted[target] == 1) & (df_test[target] == 1)).sum()
    except KeyError:
        commun_one_values = 0
    
    filename = "output.txt"
    with open(filename, "a") as f:
        print("\n\nFOR " + target + ":", file=f)
        print("Number of zeros for " + target + " in the REAL CASE:", zero_count, file=f)
        print("Number of zeros for " + target + " in the PREDICTION CASE:", pred_zero_count, file=f)
        print("Number of commun zero values for " + target + ":", commun_zero_values, file=f)
        print("\nNumber of ones for " + target + " in the REAL CASE:", one_count, file=f)
        print("Number of ones for " + target + " in the PREDICTION CASE:", pred_one_count, file=f)
        print("Number of commun one values for " + target + ":", commun_one_values, file=f)
    
def CompareWithPrediction(dfWithRealValue):
    all_subfolders = [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name)) and name.startswith("product_")]
    all_percentage = []
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
        all_percentage.append(similarity)
    
    DrawSimilarityBar(all_subfolders, all_percentage)


def CompareUsingModel(dfWithRealValue, dfToPredict):
    all_subfolders = [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name)) and name.startswith("product_")]
    all_percentage = []
    df_test = dfWithRealValue.copy()
    dataToPredict = TransformDfToPredict(dfToPredict)

    for target in all_subfolders:
        model_file_name = 'model_' + target + '.pkl'
        
        with open(os.path.join(target, model_file_name), "rb") as file:
            model = pickle.load(file)

        
        print("Predicting prediction of test for being buyed...")
        predictions = model.predict(dataToPredict)
        dfToPredict[target] = predictions
        
        
        CountingZerosAndOne(df_test, dfToPredict, target)
        
        buyed1 = dfToPredict[target]
        buyed2 = df_test[target]

        similarity = (buyed1 == buyed2).mean() 
        percentage = similarity*100
        
        print("The similarity of the "+target+" column is: ",percentage, "%")
        all_percentage.append(similarity)
    
    DrawSimilarityBar(all_subfolders, all_percentage)
    
    
dfWithRealValue= LoadCsv("Cleaned_Renamed_Compare_April2016.csv", "Cleaned_Renamed_Compare_April2016.csv")
dfToPredict = dfWithRealValue.copy()
columns_to_remove = [col for col in dfWithRealValue.columns if "product_" in col]
dfToPredict.drop(columns=columns_to_remove, inplace=True)

CompareUsingModel(dfWithRealValue, dfToPredict)
