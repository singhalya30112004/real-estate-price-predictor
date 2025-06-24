#!/opt/anaconda3/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleaned_india_real_estate.csv")

print("Basic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nCities in dataset:")
print(df['city'].value_counts())

# Create price per sqft column
df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']  # assuming price is in lakhs

# Plot price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=100, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()