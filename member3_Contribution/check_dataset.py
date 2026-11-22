import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nMissing values:")
print(train.isnull().sum())
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)

print("\nColumn names:")
for col in train.columns:
    print(col)

print("\nTarget distribution:")
print(train["SeriousDlqin2yrs"].value_counts())

print("\nTarget percentages:")
print(
    train["SeriousDlqin2yrs"]
    .value_counts(normalize=True) * 100
)