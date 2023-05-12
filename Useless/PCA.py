

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from DataCleaning import LoadCsv


# Load dataset
data = LoadCsv("Cleaned_Renamed_smalltrain_ver2.csv", "Cleaned_Renamed_smalltest_ver2.csv")
cols_to_drop = [col for col in data.columns if col.startswith('product_') and not col == "product_credit_card"]
data = data.drop(columns=cols_to_drop)

# Select only numeric features
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_data = data[numeric_cols]

# Convert non-numeric features to numeric using label encoding
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
X_non_numeric = data[non_numeric_cols]
X_non_numeric = X_non_numeric.apply(LabelEncoder().fit_transform)

# Concatenate numeric and non-numeric features
X = pd.concat([numeric_data, X_non_numeric], axis=1)

y = data["product_credit_card"]

# Apply PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize data in the new 2D space
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


'''
VERSION 1
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
data = pd.read_csv("your_dataset.csv")

# Split data into features and target
X = data.drop("target_variable", axis=1)
y = data["target_variable"]

# Apply PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize data in the new 2D space
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
'''