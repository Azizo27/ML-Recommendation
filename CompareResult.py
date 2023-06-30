import pandas as pd
from LoadCsv import LoadCsv
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from TransformDfToPredict import TransformDfToPredict
from EncodingFeatures import EncodingAllFeatures
import gzip

def DrawSimilarityBar(all_products, all_percentage):
    cmap = mcolors.LinearSegmentedColormap.from_list('CustomMap', ['blue', 'red'])
    
    fig, ax = plt.subplots()
    bars = ax.bar(all_products, all_percentage, color=cmap(all_percentage))
    
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


def CompareUsingModel(dfWithRealValue, dfToPredict, month, all_products):
    all_percentage = []
    df_test = dfWithRealValue.copy()
    dataToPredict = TransformDfToPredict(dfToPredict, month)

    for target in all_products:
            
        print("Loading the trained model from the compressed file...")
        with gzip.open(os.path.join(month, target + '.pkl.gz'), "rb") as file:
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
    
    DrawSimilarityBar(all_products, all_percentage)
    

selected_month = "May"
dfWithRealValue= LoadCsv("Cleaned_Renamed_Compare_"+selected_month+"2016.csv", "Cleaned_Renamed_Compare_"+selected_month+"2016.csv")
dfToPredict = dfWithRealValue.copy()
all_products = [col for col in dfWithRealValue.columns if "product_" in col]
dfToPredict.drop(columns=all_products, inplace=True)

CompareUsingModel(dfWithRealValue, dfToPredict, selected_month, all_products)
