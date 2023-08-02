
from calculations import PredictUsingHistorical

import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from LoadCsv import LoadCsv


if __name__ == '__main__':
    
    df= LoadCsv("Cleaned_Renamed_train_ver2.csv", "Cleaned_Renamed_train_ver2.csv")
    all_columns = [ "product_savings_account", "product_current_accounts"]
    
    final_df = PredictUsingHistorical(df, all_columns)
    
    # Get the path to the "prediction" subfolder
    prediction_subfolder = os.path.join(project_dir, "prediction")
    
    # Create the full file path for the output CSV file inside the "prediction" subfolder
    output_csv_path = os.path.join(prediction_subfolder, 'predictions.csv')
    
    final_df.to_csv(output_csv_path, index=False)
