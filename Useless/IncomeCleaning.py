import pandas as pd
import numpy as np

def replace_null_gross_income(df):
    print("Extracting relevant columns")
    columns_of_interest = ['customer_code', 'gross_income']
    product_columns = [col for col in df.columns if col.startswith('product_')]
    df_filtered = df[columns_of_interest + product_columns].copy()

    print("Calculating median income for each combination of products")
    grouped = df_filtered.groupby(product_columns)['gross_income'].median().reset_index()
    grouped.rename(columns={'gross_income': 'median_income'}, inplace=True)
    
    print("Calculating mean income for each combination of products")
    grouped_mean = df_filtered.groupby(product_columns)['gross_income'].mean().reset_index()
    grouped_mean.rename(columns={'gross_income': 'mean_income'}, inplace=True)

    print("Creating a dictionary to map each customer code to the corresponding median income")
    customer_income_mapping = {}
    for _, row in grouped.iterrows():
        product_combination = tuple(row[product_columns])
        median_income = row['median_income']
        customer_codes = df_filtered.loc[df_filtered[product_columns].eq(product_combination).all(axis=1), 'customer_code']
        customer_income_mapping.update({code: median_income for code in customer_codes})

    print("Creating a dictionary to map each customer code to the corresponding mean income")
    customer_income_mapping_mean = {}
    for _, row in grouped_mean.iterrows():
        product_combination = tuple(row[product_columns])
        mean_income = row['mean_income']
        customer_codes = df_filtered.loc[df_filtered[product_columns].eq(product_combination).all(axis=1), 'customer_code']
        customer_income_mapping_mean.update({code: mean_income for code in customer_codes})
        
    '''
    print("Replacing null values with corresponding median incomes")
    df['gross_income'] = df['gross_income'].fillna(df['customer_code'].map(customer_income_mapping))
    df = df.dropna(subset=['gross_income'])
    '''
    
    print("Creating new column 'median_income' and filling it with corresponding median incomes")
    df['median_income'] = df['customer_code'].map(customer_income_mapping)
    df = df.dropna(subset=['median_income'])
    
    
    print("Creating new column 'mean_income' and filling it with corresponding mean incomes")   
    df['mean_income'] = df['customer_code'].map(customer_income_mapping_mean)
    df = df.dropna(subset=['mean_income'])

    return df



def replace_null_gross_incomeUsingStartDate(df):
    df['customer_start_date'] = pd.to_datetime(df['customer_start_date'])
    df['customer_start_month'] = df['customer_start_date'].dt.to_period('M')
    
    df['median_income'] = df.groupby('customer_start_date')['gross_income'].transform('median')
    df['median_income_month'] = df.groupby('customer_start_month')['gross_income'].transform('median')
    
    df['mean_income'] = df.groupby('customer_start_date')['gross_income'].transform('mean')
    df['mean_income_month'] = df.groupby('customer_start_month')['gross_income'].transform('mean')

    # Print the updated DataFrame
    print(df[['date', 'customer_start_date', 'gross_income', 'median_income', 'median_income_month','mean_income', 'mean_income_month']].head(10))
    return df

