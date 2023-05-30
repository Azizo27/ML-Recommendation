import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


def SelectFeaturesAndOutput(df, product_name):
    
    # To delete all product columns except the one we want to predict
    cols_to_drop = [col for col in df.columns if col.startswith('product_') and not col == product_name]
    df = df.drop(columns=cols_to_drop)
    '''
            0   date                                          object 
            1   customer_code                                 int64  
            2   employee_index                                object 
            3   country_residence                             object 
            4   gender                                        object 
            5   age                                           int64  
            6   customer_start_date                           object     
            7   new_customer_index                            int64  
            8   customer_seniority                            int64
            9   primary_customer_index                        int64
            10  last_date_as_primary_customer                 object  PAS EXPLOITABLE (NULL VALUES) (TO BE DROPPED)
            11  customer_type_at_beginning_of_month           object
            12  customer_relation_type_at_beginning_of_month  object
            13  residence_index                               object
            14  foreigner_index                               object
            15  spouse_index                                  object
            16  channel_used_by_customer_to_join              object
            17  deceased_index                                object
            18  address_type                                  int64
            19  province_code                                 int64
            20  province_name                                 object
            21  activity_index                                int64
            22  gross_income                                  float64
            23  segmentation                                  object
            24  product_credit_card                           int64
    '''
    
    return df


def PredictionProduct(df, product_name):
    # Load the cleaned dataset
    data = pd.read_csv("cleaned_dataset.csv")

    # Feature Engineering
    # Convert categorical variables into numerical ones using LabelEncoder
    categorical_columns = ['employee_index', 'country_residence', 'gender', 'residence_index', 'foreigner_index', 'spouse_index', 'channel_used_by_customer_to_join', 'deceased_index', 'province_name', 'segmentation']
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    # Normalize or scale numerical features
    numerical_columns = ['age', 'customer_seniority', 'gross_income']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Split the data
    X = data.drop("product_credit_card", axis=1)
    y = data["product_credit_card"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select an algorithm
    clf = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Train the model with the best hyperparameters
    best_params = random_search.best_params_
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC AUC score:", roc_auc_score(y_test, y_pred))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature Importance
    feature_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importances:\n", feature_importances)
