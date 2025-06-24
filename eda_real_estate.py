#!/opt/anaconda3/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleaned_india_real_estate.csv")

# Drop non-useful column
df.drop(columns=['furnishing'], inplace=True)

print(df['price'].head(10))

print("Basic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nCities in dataset:")
print(df['city'].value_counts())

# Create Price Per sqft Column
df['price_per_sqft'] = (df['price']) / df['total_sqft']


# Plot Price Distribution
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


# Furnishing Type vs Price Boxplot
# Result: No significant impact on price, so dropped from features
# scatter_df = df[(df['total_sqft'] < 5000) & (df['price'] < 5e7)]  # price < ₹5 Cr
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='furnishing', y='price', data=scatter_df)
# plt.ylim(0, 5e7)
# plt.title("Furnishing Type vs Price")
# plt.xlabel("Furnishing Type")
# plt.ylabel("Price (₹ Crores)")
# plt.tight_layout()
# plt.show()


# Correlation Heatmap for Numerical Features
import numpy as np

corr_df = df[['BHK', 'bathroom', 'balcony', 'total_sqft', 'parking', 'price']].copy()
corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
corr_df.dropna(inplace=True)
corr_matrix = corr_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()

# Drop weak predictors before continuing
df.drop(columns=['balcony', 'parking'], inplace=True)


# Top Localities by Price per Sqft

location_city_stats = df.groupby(['location', 'city'])['price_per_sqft'].mean().reset_index()
top10 = location_city_stats.sort_values(by='price_per_sqft', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='price_per_sqft', y='location', hue='city', data=top10, dodge=False, palette='Set2')
plt.title("Top 10 Most Expensive Locations (₹ per sqft) - Colored by City")
plt.xlabel("Price per Sqft (₹)")
plt.ylabel("Location")
plt.legend(title='City')
plt.tight_layout()
plt.show()