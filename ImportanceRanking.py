

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from DataCleaning import LoadCsv, DisplayInformation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
def PredictionProduct(data, target):
    print("Starting...")
    target = "product_credit_card"
    product_to_drop = [col for col in data.columns if col.startswith('product_') and not col == target]
    feature_to_drop = ["date", "customer_code", "last_date_as_primary_customer", "province_name", "province_code", 
                       "segmentation", "customer_start_date", "channel_used_by_customer_to_join", "country_residence", "spouse_index",
                       "deceased_index", "employee_index", "address_type", "foreigner_index", "primary_customer_index", "new_customer_index"]
    data = data.drop(columns=product_to_drop)
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
    print("Importances...")
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]

    # Print feature importances in descending order
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {importances[idx]}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


data = LoadCsv("Cleaned_Renamed_biggertrain_ver2.csv", "Cleaned_Renamed_smalltest_ver2.csv")
all_products=  [col for col in data.columns if col.startswith('product_')]

for product in all_products:
    PredictionProduct(data, product)
    print("Done with", product)
