import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


def predict_income(df):

    selected_columns = ['age', 'gender', 'province_code', 'gross_income']
    data = df[df['gross_income'].notnull()][selected_columns]

    # Preprocess the data
    # One-hot encode categorical features (gender)
    print('One-hot encoding categorical features...')
    categorical_features = ['gender']
    one_hot_encoder = OneHotEncoder()
    column_transformer = ColumnTransformer([('encoder', one_hot_encoder, categorical_features)], remainder='passthrough')

    # Remove the target variable (gross_income) from the features
    print('Removing the target variable...')
    X = data.drop(columns=['gross_income'])
    y = data['gross_income']

    # Apply one-hot encoding and scaling
    print('Applying one-hot encoding and scaling...')
    X_encoded = column_transformer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Split the data into training and testing sets
    print('Splitting the data into training and testing sets...')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    print('Training the linear regression model...')
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    print('Making predictions on the test set...')
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    print('Calculating the mean squared error...')
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error: {mse}')
    
    # Save the trained model to a file
    print('Saving the trained model to a file...')
    with open('gross_income_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    '''
    # Load the saved model from the file
    with open('gross_income_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Use the loaded model to make predictions
    y_pred = loaded_model.predict(X_test)
    '''
