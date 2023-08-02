

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