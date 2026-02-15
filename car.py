# =========================================================
# APPLIED DATA SCIENCE
# Assignment 1 – Python for Data Handling
# Dataset: Car Sales & Specifications Dataset
# =========================================================

# ---------------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("car_data.csv")
print("Dataset Loaded Successfully")
print("=" * 100)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()
print("Column Names After Cleaning:")
print(df.columns.tolist())
print("=" * 100)

# ---------------------------------------------------------
# 4. Dataset Exploration
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())
print("=" * 100)

print("Dataset Shape:", df.shape)
print("=" * 100)

print("Dataset Information:")
df.info()
print("=" * 100)

print("Statistical Summary:")
print(df.describe())
print("=" * 100)

# ---------------------------------------------------------
# 5. Check Missing Values
# ---------------------------------------------------------
print("Missing Values:")
print(df.isnull().sum())
print("=" * 100)

# ---------------------------------------------------------
# 6. Check Zero Values in Numerical Columns
# ---------------------------------------------------------
numerical_cols = df.select_dtypes(include=np.number).columns
print("Zero Values in Numerical Columns:")
print((df[numerical_cols] == 0).sum())
print("=" * 100)

# ---------------------------------------------------------
# 7. Remove Duplicate Records
# ---------------------------------------------------------
print("Duplicate Records:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates Removed")
print("=" * 100)

# ---------------------------------------------------------
# 8. Handle Missing Values
# ---------------------------------------------------------
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled")
print("=" * 100)

# ---------------------------------------------------------
# 9. Feature Engineering – Create Car Age Column
# ---------------------------------------------------------
df['car_age'] = 2024 - df['year']
print("Car Age Column Created Successfully")
print(df[['year', 'car_age']].head())
print("=" * 100)

# ---------------------------------------------------------
# 10. Statistical Measures for Selling Price
# ---------------------------------------------------------
print("Selling Price Statistics")
print("Mean:", df['selling_price'].mean())
print("Median:", df['selling_price'].median())
print("Mode:", df['selling_price'].mode()[0])
print("Skewness:", df['selling_price'].skew())
print("=" * 100)

# ---------------------------------------------------------
# 11. Basic Visualizations
# ---------------------------------------------------------

# Histogram of Selling Price
plt.figure(figsize=(6,4))
sns.histplot(df['selling_price'], kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.show()

# Count Plot of Fuel Type
plt.figure(figsize=(6,4))
sns.countplot(x='fuel_type', data=df)
plt.title("Fuel Type Distribution")
plt.show()

# Scatter Plot: Present Price vs Selling Price
plt.figure(figsize=(6,4))
sns.scatterplot(x='present_price', y='selling_price', data=df)
plt.title("Present Price vs Selling Price")
plt.show()

print("Preprocessing and Analysis Completed Successfully")
