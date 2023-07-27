import pandas as pd
import numpy as np
from LoadCsv import LoadCsv
import time

def exponential_smoothing_with_date(series, alpha):
    result = np.empty(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        weight = alpha ** (len(series) - i)
        result[i] = weight * series[i] + (1 - weight) * result[i-1]
    return np.round(result, 3)

def apply_exponential_smoothing(df_chunk, columns, alpha):
    final_rows = []
    for _, group in df_chunk.groupby('customer_code'):
        smoothed_values = {}
        for col in columns:
            smoothed_values[col] = exponential_smoothing_with_date(group[col].values, alpha)[-1]
        final_rows.append({
            'customer_code': group['customer_code'].iloc[0],
            **smoothed_values
        })
    return pd.DataFrame(final_rows)

# Custom chunking function to ensure complete groups are included in each chunk
def custom_chunking(df, chunk_size):
    idx = 0
    while idx < len(df):
        group_end = idx + chunk_size
        if group_end >= len(df):
            yield df.iloc[idx:]
            break
        while group_end < len(df) and df['customer_code'].iloc[group_end] == df['customer_code'].iloc[idx]:
            group_end += 1
        yield df.iloc[idx:group_end]
        idx = group_end




def PredictUsingHistorical(df, all_columns):

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

    final_df.to_csv('pred_final.csv', index=False)


df= LoadCsv("Cleaned_Renamed_train_ver2.csv", "Cleaned_Renamed_train_ver2.csv")
all_columns = [ "product_savings_account", "product_guarantees", "product_current_accounts",
        "product_derivada_account", "product_payroll_account", "product_junior_account",
        "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
        "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
        "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
        "product_loans", "product_taxes", "product_credit_card", "product_securities",
        "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]
PredictUsingHistorical(df, all_columns)