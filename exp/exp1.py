import pandas as pd

df1 = pd.read_csv("titanic.csv")

print(df1.head())
df=df1["Age"]  #To find for only one column


# Central Tendency
print("Mean\n")
print(df.mean(numeric_only=True))
print("median\n")
print(df.median(numeric_only=True))
print("mode\n")

print(df.mode().iloc[0])

# Dispersion
print("variance\n")

print(df.var(numeric_only=True))
print("Stand Dev\n")

print(df.std(numeric_only=True))
print("Range\n")

print(df.max(numeric_only=True) - df.min(numeric_only=True))
print("Skew\n")


# Shape
print(df.skew(numeric_only=True))
print("kurtosis\n")

print(df.kurt(numeric_only=True))

print("new\n")
print(df.describe())