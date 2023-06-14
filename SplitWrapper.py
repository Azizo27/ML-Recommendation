from sklearn.model_selection import train_test_split
from EncodingFeatures import EncodingAllFeatures

first_time = True

def splitting_dataset(df, features, target, size, rs):
    # Check if this is the first time the function is called
    global first_time
    if first_time == True:
        print ("The split has not been performed before.")
        Features_filename = 'FittedFeaturesofModels.txt'
        X = EncodingAllFeatures(df[features])
        with open(Features_filename,'w') as file:
            column_names = X.columns.tolist()
            file.write(','.join(str(item) for item in column_names))
     
        y = df[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=rs)
        
        # Save the initial feature split for future use
        globals()['X_train'] = X_train
        globals()['X_test'] = X_test
        first_time = False
    else:
        # Use the existing feature split
        X_train = globals()['X_train']
        X_test = globals()['X_test']
        
        # Map the target variable based on the corresponding rows in X_train
        y_train = df.loc[X_train.index, target]
        y_test = df.loc[X_test.index, target]

    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
