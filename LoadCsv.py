from RenameColumns import RenameColumns
from CleaningData import CleaningData
import pandas as pd
import numpy as np
import os
import calendar
from DisplayInformation import DisplayInformation


#This function takes two parameters (file_name and new_file_name) to create a new cleaned/renamed csv file. then, it returns it as a dataframe.
#In our case, file_name is the original train dataset 'train_ver2.csv',
# While new_file_name is the name we want to give to the cleaned/Renamed csv file that will be created.

#If both files already exist, the function will return the dataframe from the csv file that correspond to file_name.

def LoadCsv(file_name, new_file_name):
    chunksize = 100000
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, low_memory=False, chunksize=chunksize):
        chunks.append(chunk)

    print("Data loaded. Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    if not os.path.exists(new_file_name):
        print("Renaming columns...")
        df = RenameColumns(df)
        print("Cleaning data...")
        df = CleaningData(df)
        print("Saving to a new csv file...")
        df.to_csv(new_file_name, index=False)
        
    return df


# The file_name must be the original train dataset BUT Cleaned and renamed
def LoadPerMonth(file_name, month_name):
    month_number = str(list(calendar.month_name).index(month_name)).zfill(2)
    chunksize = 100000
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, low_memory=False, chunksize=chunksize):
        filtered_chunk = chunk[chunk['date'].astype(str).str.startswith('2015-{}-'.format(month_number))]
        chunks.append(filtered_chunk)

    print("Data loaded. Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    return df 
