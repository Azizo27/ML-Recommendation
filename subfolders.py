import os
import pandas as pd

def merge_dataframes():
    subfolders = [f for f in os.listdir('.') if f.startswith('product_') and os.path.isdir(f)]

    dfs = []
    for subfolder in subfolders:
        csv_files = [f for f in os.listdir(subfolder) if f.startswith('prediction_') and f.endswith('.csv')]

        
        for csv_file in csv_files:
            file_path = os.path.join(subfolder, csv_file)

            df = pd.read_csv(file_path)
            dfs.append(df)
            
    merged_df = pd.concat(dfs, ignore_index=False)
    
    final_df = merged_df.groupby(merged_df.index).first().reset_index(drop=True)

    final_df.to_csv("Finalprediction.csv", index=False)


merge_dataframes()