
from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
print("🔹 Features shape:", X.shape)
print("🔹 Target classes:", data.target_names.tolist())
print("🔹 First 5 rows of features:\n", X.head())
print("🔹 Target distribution:\n", y.value_counts())
