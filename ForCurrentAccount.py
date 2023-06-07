import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from LoadCsv import LoadCsv

def create_classification_model(features, target, dataframe):
    # Select the features and target variable from the original dataframe
    df = dataframe[features + [target]].copy()

    # Preprocessing
    # Encode categorical variables if present
    categorical_cols = ['segmentation', 'gender', 'customer_relation_type_at_beginning_of_month']
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Perform feature scaling on numerical variables
    numerical_cols = ['age', 'gross_income', 'customer_seniority']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split the data into training and testing sets
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(df[target].unique()), activation='softmax'))  # Assuming a multi-class classification problem

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model

    


features = ['age', 'gross_income', 'customer_seniority', 'customer_relation_type_at_beginning_of_month', 'segmentation', 'gender']
dfForTraining = LoadCsv("Cleaned_Renamed_train_May2015.csv", "Cleaned_Renamed_train_May2015.csv")

create_classification_model(features, 'product_current_accounts', dfForTraining)