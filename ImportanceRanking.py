

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from DataCleaning import LoadCsv, DisplayInformation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
def PredictionProduct(df, target):
    print("Starting...")
    product_to_drop = [col for col in df.columns if col.startswith('product_') and not col == target]
    feature_to_drop = ["date", "customer_code", "last_date_as_primary_customer", "province_name", "province_code", 
                       "customer_start_date", "channel_used_by_customer_to_join", "country_residence", "spouse_index",
                       "deceased_index", "employee_index", "address_type", "foreigner_index", "primary_customer_index", "new_customer_index"]
    data = df.drop(columns=product_to_drop)
    data = data.drop(columns=feature_to_drop)

    # Split data into numeric features and target
    print("Splitting Numerical and Categorical Features...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = data[numeric_cols]

    # Convert non-numeric features to numeric using one-hot encoding
    print("Converting...")
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    X_non_numeric = pd.get_dummies(data[non_numeric_cols])

    # Concatenate numeric and non-numeric features
    print("Concatenating And Splitting...")
    X = pd.concat([X_numeric, X_non_numeric], axis=1)
    X = X.drop(target, axis=1)

    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    
    # Get feature importances and sort them in descending order
    print("Importances of each features...")
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {importances[idx]}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Use predict_proba to get the probability
    
    print("Predicting probability of test for being buyed...")
    probabilities = model.predict_proba(X_test)
    df2 = pd.DataFrame(data=X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    # SHOULD ALWAYS BE > 1 BECAUSE IF NOT, IT MEANS THAT THE ACCURACY IS 100% AND THIS IS NOT POSSIBLE WITH A BIG DATASET 
    # (Because It would mean that in the entire dataset, there is not a single client who bought the product)
    if probabilities.shape[1] > 1:
        df2['probability_buyed'] = probabilities[:, 1]
    else:
        # If there is only one column, assign (1 - probabilities[:, 0]) to 'probability_buyed'
        #With this operation, probability_buyed will be equal to 1 if the client bought the product and 0 if he didn't
        df2['probability_buyed'] = 1 - probabilities[:, 0]
    
    
    df_non_zero = df2[df2['probability_buyed'] >  0.5]
    print(df_non_zero)


df = LoadCsv("Cleaned_Renamed_smalltrain_ver2.csv", "Cleaned_Renamed_smalltest_ver2.csv")
all_products=  [col for col in df.columns if col.startswith('product_')]

for product in all_products:
    PredictionProduct(df, product)
    print("Done with", product)