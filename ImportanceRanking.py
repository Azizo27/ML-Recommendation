

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from DataCleaning import LoadCsv, DisplayInformation


# Load dataset
data = LoadCsv("Cleaned_Renamed_JuneOnly.csv", "Cleaned_Renamed_smalltest_ver2.csv")
print("Starting...")
target = "product_credit_card"
product_to_drop = [col for col in data.columns if col.startswith('product_') and not col == target]
feature_to_drop = ["date", "customer_code", "last_date_as_primary_customer", "province_name", "province_code", "segmentation", "customer_start_date", "channel_used_by_customer_to_join"]
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

print("Fitting...")
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances and sort them in descending order
print("Importances...")
importances = rf.feature_importances_
sorted_indices = importances.argsort()[::-1]

# Print feature importances in descending order
for idx in sorted_indices:
    print(f"{X.columns[idx]}: {importances[idx]}")




'''
VERSION 1

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = pd.read_csv()

# Split data into features and target
X = data.drop("target_variable", axis=1)
y = data["target_variable"]

# Fit Random Forest model to the data
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances and sort them in descending order
importances = rf.feature_importances_
sorted_indices = importances.argsort()[::-1]

# Print feature importances in descending order
for idx in sorted_indices:
    print(f"{X.columns[idx]}: {importances[idx]}")

'''