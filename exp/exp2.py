import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.info())

# Missing values
df = df.fillna(df.mean(numeric_only=True))

# Normalization
scaler = MinMaxScaler()
df[df.select_dtypes(include='number').columns] = scaler.fit_transform(
    df.select_dtypes(include='number')
)

# Scaling
scaler = StandardScaler()
df[df.select_dtypes(include='number').columns] = scaler.fit_transform(
    df.select_dtypes(include='number')
)

# Visualization

# Histogram
df.hist(figsize=(10,8))
plt.show()

# Boxplot
sns.boxplot(data=df)
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# ✅ Scatter Plot (ADDED)
numeric_cols = df.select_dtypes(include='number').columns

if len(numeric_cols) >= 2:
    sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title("Scatter Plot")
    plt.show()
else:
    print("Not enough numeric columns for scatter plot")