from RenameColumns import RenameColumns
from CleaningData import CleaningData
import pandas as pd
import numpy as np
import os

#This function loads the data ENTIRELY from a csv. Also, It can be filtered by date May/June. Also, It can copy to a new csv file by renaming the columns
def LoadCsv(file_name, new_file_name):
    chunksize = 100000
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, low_memory=False, chunksize=chunksize):
        '''
        #TO FILTER BY DATE (NB: YOU MUST DELETE THE NEXT LINE)
        filtered_chunk = chunk[chunk['fecha_dato'].astype(str).str.startswith(('2015-06'))]
        chunks.append(filtered_chunk)
        '''
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
