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
sns.histplot(df['price'] / 1e5, bins=100, kde=True)
plt.xlim(0, 500)  # Limit to 0–500 lakhs (₹0–₹5Cr)
plt.title("Price Distribution (Zoomed In)")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Price Per Sqft Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price_per_sqft'], bins=100, kde=True)
plt.xlim(0, 20000)  # Focus on most realistic values (₹0–₹20k/sqft)
plt.title("Price per Sqft Distribution (Zoomed In)")
plt.xlabel("Price per Sqft (₹)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Scatterplot: Area vs Price
# Filtering to remove extreme outliers
scatter_df = df[(df['total_sqft'] < 5000) & (df['price'] < 5e7)]  # price < ₹5 Cr
scatter_df['price_cr'] = scatter_df['price'] / 1e7
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_sqft', y='price_cr', data=scatter_df, alpha=0.4)
plt.title("Total Sqft vs Price")
plt.xlabel("Total Sqft")
plt.ylabel("Price (₹ Crores)")
plt.tight_layout()
plt.show()


# City-wise Avg Price per Sqft
city_avg = df.groupby('city')['price_per_sqft'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=city_avg.index.str.title(), y=city_avg.values, palette="Set2")
plt.title("Average Price per Sqft by City")
plt.xlabel("City")
plt.ylabel("Avg Price per Sqft (₹)")
plt.tight_layout()
plt.show()


# BHK vs Price Boxplot
scatter_df = df[(df['total_sqft'] < 5000) & (df['price'] < 5e7)]  # price < ₹5 Cr
plt.figure(figsize=(10, 5))
sns.boxplot(x='BHK', y='price', data=scatter_df)
plt.ylim(0, 5e7)  # ₹5 Cr, adjust based on your data
plt.title("BHK vs Price")
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Price (₹ Crores)")
plt.tight_layout()
plt.show()