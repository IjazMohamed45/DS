import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("titanic.csv")

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical to numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================
# ✅ 1. LINEAR REGRESSION
# ==================================================

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

# Convert predictions to 0/1
y_pred_lin = [1 if val > 0.5 else 0 for val in y_pred_lin]

print("\n--- Linear Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lin))
print("MSE:", mean_squared_error(y_test, y_pred_lin))

# ==================================================
# ✅ 2. LOGISTIC REGRESSION
# ==================================================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))