import pandas as pd
import numpy as np
import time
from chunking import custom_chunking



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

    return final_df

