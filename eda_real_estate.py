#!/opt/anaconda3/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleaned_india_real_estate.csv")

print(df['price'].head(10))

print("Basic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nCities in dataset:")
print(df['city'].value_counts())

# Create price per sqft column
df['price_per_sqft'] = (df['price']) / df['total_sqft']

# Plot price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price'] / 100000, bins=100, kde=True)
plt.xlim(0, 500)  # Limit to 0–500 lakhs (₹0–₹5Cr)
plt.title("Price Distribution (Zoomed In)")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()