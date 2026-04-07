import pandas as py

df=py.read_csv("titanic.csv")

df1=df["Age"]

print("Mean:\n")

print(df1.mean(numeric_only=True))
print("Median:\n")
print(df1.median(numeric_only=True))
print("Mode:\n")
print(df1.mode().iloc[0])
print("Variance:\n")
print(df1.var(numeric_only=True))
print("Standard Deviation:\n")
print(df1.std(numeric_only=True))
print("Range:\n")
print(df1.max(numeric_only=True) - df1.min(numeric_only=True))
print("Skewness:\n")
print(df1.skew(numeric_only=True))
print("Kurtosis:\n")
print(df1.kurt(numeric_only=True))


