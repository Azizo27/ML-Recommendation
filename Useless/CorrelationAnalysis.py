import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset
data = pd.read_csv("your_dataset.csv")

# Select only numeric features
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_data = data[numeric_cols]

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Visualize correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True)
