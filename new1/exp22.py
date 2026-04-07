import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.info())

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# -------------------------------
# SPLITTING (Assuming last column is target)
# -------------------------------
X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# NORMALIZATION (ONLY on TRAIN)
# -------------------------------
cat_cols=['sex','name'] 
num_cols = X_train.select_dtypes(include='number').columns.difference(cat_cols)

minmax = MinMaxScaler()
X_train[num_cols] = minmax.fit_transform(X_train[num_cols])
X_test[num_cols] = minmax.transform(X_test[num_cols])

# -------------------------------
# STANDARDIZATION (ONLY on TRAIN)
# -------------------------------
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------
# VISUALIZATION (use full dataset OR train set)
# -------------------------------

# Histogram
X_train.hist(figsize=(10,8))
plt.show()

# Boxplot
sns.boxplot(data=X_train)
plt.show()

# Heatmap
sns.heatmap(X_train.corr(numeric_only=True), annot=True)
plt.show()

# Scatter Plot
if len(num_cols) >= 2:
    sns.scatterplot(x=X_train[num_cols[0]], y=X_train[num_cols[1]])
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.title("Scatter Plot")
    plt.show()
else:
    print("Not enough numeric columns for scatter plot")