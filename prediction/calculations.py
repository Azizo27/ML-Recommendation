import pandas as pd
import numpy as np


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
