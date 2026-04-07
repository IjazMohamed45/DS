import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

print("First 5 rows:\n", df.head())

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# -------------------------------
# COVARIANCE
# -------------------------------
cov_matrix = numeric_df.cov()
print("\nCovariance Matrix:\n", cov_matrix)

# -------------------------------
# CORRELATION
# -------------------------------
corr_matrix = numeric_df.corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# -------------------------------
# HEATMAP FOR CORRELATION
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# HEATMAP FOR COVARIANCE
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(cov_matrix, annot=True, cmap='viridis')
plt.title("Covariance Heatmap")
plt.show()