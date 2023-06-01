
from LoadCsv import LoadCsv
import pandas as pd


def RankRecommendation(df):
    final_df =  pd.DataFrame(df['customer_code'])
    df= df.drop('customer_code', axis=1)
    # Create an empty list to store the results
    results = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Extract the values from the row and create an array of dictionaries
        row_values = row.to_dict()
        result = [{col: value} for col, value in row_values.items()]
        # Sort the array based on the 'valueofproduct' in descending order
        result = sorted(result, key=lambda x: list(x.values())[0], reverse=True)
        # Append the sorted array to the results list
        results.append(result)

    # Create a new DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results)

    # Rename the columns as "Recommendation_1", "Recommendation_2", etc.
    results_df.columns = [f"Recommendation_{i+1}" for i in range(len(results_df.columns))]

    for col in results_df.columns:
        final_df[col] = results_df[col]
        
    final_df.to_csv("Finaldf.csv", index=False)