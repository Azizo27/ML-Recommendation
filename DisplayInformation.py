import pandas as pd
import numpy as np

#This function displays the information of the data
def DisplayInformation(df):
    print("Info:", df.info())
    # Display unique values for each column
    for col in df.columns:
        unique_values = df[col].unique()
        print("Unique values in the '{}' column:".format(col))
        null_count = df[col].isnull().sum()
        print("Number of null values in the column ",col," :  ",null_count)
        print(unique_values)
