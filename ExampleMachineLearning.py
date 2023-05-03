import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Preprocess the categorical features
categorical_columns = ['employee_index', 'country_residence', 'gender', 'new_customer_index',
                      'primary_customer_index', 'customer_type_at_beginning_of_month', 
                      'customer_relation_type_at_beginning_of_month', 'residence_index', 
                      'foreigner_index', 'spouse_index', 'channel_used_by_customer_to_join',
                      'deceased_index', 'address_type', 'province_code', 'province_name', 'activity_index']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))

# Convert date columns to datetime and extract year and month as features
date_columns = ['date', 'customer_start_date', 'last_date_as_primary_customer']
for column in date_columns:
    data[column] = pd.to_datetime(data[column])
    data[f'{column}_year'] = data[column].dt.year
    data[f'{column}_month'] = data[column].dt.month
    data = data.drop(column, axis=1)

# Fill missing values
data['age'].fillna(data['age'].median(), inplace=True)
data['customer_seniority'].fillna(data['customer_seniority'].median(), inplace=True)
data['gross_income'].fillna(data['gross_income'].median(), inplace=True)

# Split the data into training and testing sets
X = data.drop('product_credit_card', axis=1)
y = data['product_credit_card']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the continuous features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
