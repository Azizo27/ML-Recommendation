import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def NeuralNetworkGrossIncome(data):

    # Split the data into train and test sets
    selected_columns = ['age', 'gender', 'province_name', 'gross_income']
    df = data[data['gross_income'].notnull()][selected_columns]
    
    X = df.drop("gross_income", axis=1)
    y = df["gross_income"]
    
    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Preprocessing
    numeric_features = ["age"]
    categorical_features = ["gender", "province_name"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features)
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Neural network model
    input_dim = X_train_processed.shape[1]
    output_dim = 1

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(output_dim)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    history = model.fit(X_train_processed, y_train, epochs=100, batch_size=32, validation_data=(X_val_processed, y_val))

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test_processed, y_test)

    '''
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    '''
    
    #Save the model
    model.save("my_model.h5")
    
    '''
    # Load the model
    loaded_model = tf.keras.models.load_model("my_model.h5")

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X_test_processed)

    # Example of using the loaded model
    sample_input = X_test_processed[0].reshape(1, -1)  # Reshape the first test sample to have the required input shape
    sample_prediction = loaded_model.predict(sample_input)
    print("Sample prediction:", sample_prediction)
    '''

    print("Test Mean Absolute Error:", test_mae)


