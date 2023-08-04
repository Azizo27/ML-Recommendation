
from calculations import apply_exponential_smoothing
from chunking import custom_chunking
import pandas as pd
import time
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from LoadCsv import LoadCsv


if __name__ == '__main__':
    
    df= LoadCsv("Cleaned_Renamed_train_ver2.csv", "Cleaned_Renamed_train_ver2.csv")
    
    all_columns = [ "product_savings_account", "product_guarantees", "product_current_accounts",
        "product_derivada_account", "product_payroll_account", "product_junior_account",
        "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
        "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
        "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
        "product_loans", "product_taxes", "product_credit_card", "product_securities",
        "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]
    
    print("Doing sorting...")
    df_sorted = df.sort_values(by=['customer_code', 'date'])

    start_time = time.time()

    alpha = 0.7  # You can adjust this value based on your preference
    chunk_size = 100000  # Adjust the chunk size based on your memory capacity

    final_df_list = []


    # Process the DataFrame in custom chunks
    for chunk_df in custom_chunking(df_sorted, chunk_size):
        final_df_list.append(apply_exponential_smoothing(chunk_df, all_columns, alpha))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the program: {elapsed_time:.4f} seconds")

    # Merge the results based on customer_code
    final_df = pd.concat(final_df_list, ignore_index=True)
    final_df = final_df.drop_duplicates(subset='customer_code')
    
    
    
    # Get the path to the "prediction" subfolder
    prediction_subfolder = os.path.join(project_dir, "prediction")
    
    # Create the full file path for the output CSV file inside the "prediction" subfolder
    output_csv_path = os.path.join(prediction_subfolder, 'predictions.csv')
    
    final_df.to_csv(output_csv_path, index=False)
