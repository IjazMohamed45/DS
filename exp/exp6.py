import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("titanic.csv")

# -------------------------------
# PREPROCESSING
# -------------------------------

# Handle missing values (numeric only)
df = df.fillna(df.mean(numeric_only=True))

# Keep only numerical columns
df_num = df.select_dtypes(include='number')

# -------------------------------
# SCALING (VERY IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)


# -------------------------------
# FIND OPTIMAL K (ELBOW METHOD)
# -------------------------------
"""
wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()



"""



# -------------------------------
# APPLY K-MEANS (choose K=3 or from elbow)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)


print("Cluster Labels:\n", labels[:10])

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()