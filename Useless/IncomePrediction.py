import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

def predict_income(df):

    print('Preprocessing the data...')
    selected_columns = ['age', 'province_code', 'segmentation', 'gross_income']
    data = df[df['gross_income'].notnull() & df['segmentation'].notnull()][selected_columns]

    # One-hot encoding categorical features
    categorical_features = ['segmentation']
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = one_hot_encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_features), index=data.index)

    # Merge encoded features with the rest of the data
    data = data.drop(columns=categorical_features).join(encoded_df)

    print('Removing the target variable...')
    X = data.drop(columns=['gross_income'])
    y = data['gross_income']

    print('Splitting the data into training and testing sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training the linear regression model...')
    model = LinearRegression()
    model.fit(X_train, y_train)

    print('Making predictions on the test set...')
    y_pred = model.predict(X_test)

    print('Calculating the mean squared error...')
    mse = r2_score(y_test, y_pred)
    print(f'r2 score: {mse}')

    '''
    print('Saving the trained model to a file...')
    with open('gross_income_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    '''

    '''
    # Load the saved model from the file
    with open('gross_income_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Use the loaded model to make predictions
    y_pred = loaded_model.predict(X_test)
    '''


