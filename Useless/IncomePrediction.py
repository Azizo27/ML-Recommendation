import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


def predict_income(df):

    target = 'gross_income'
    data = df[[ 'median_income', 'mean_income', target]].copy()
    #data = df[df['gross_income'].notnull() & df['segmentation'].notnull()][selected_columns].copy()
    
    '''
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
    '''
    # FOR ONLY NUMERICAL VALUE
    X = data.drop(target, axis=1)
    y = data[target]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training the model...')
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    print('Making predictions on the test set...')
    predicted_income_test = model.predict(X_test)
    
    # Get feature importances and sort them in descending order
    print("Importances of each features...")
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    
    for idx in sorted_indices:
        print(f"{X.columns[idx]}: {importances[idx]}")

    print('Evaluating the model...')
    r2 = r2_score(y_test, predicted_income_test)
    print(f'R2 score: {r2}')

    return df
