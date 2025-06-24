#!/opt/anaconda3/bin/python

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("india_real_estate_prices.csv", on_bad_lines='skip')
print("Original Shape:", df.shape)

# Column-Wise Cleaning
# 1. Standardize text columns
df['city'] = df['city'].str.strip().str.lower()
df['location'] = df['location'].str.strip().str.lower()
df['locality_type'] = df['locality_type'].fillna("unknown").str.strip().str.lower()
df['furnishing'] = df['furnishing'].fillna("unknown").str.strip().str.lower()

# 2. BHK, bathroom, balcony, parking → numeric
for col in ['BHK', 'bathroom', 'balcony', 'parking']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 3. total_sqft: handle ranges like '1000-1200'
def convert_sqft(x):
    try:
        if isinstance(x, str):
            if '-' in x:
                parts = x.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            else:
                return float(x)
        return x
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())
df = df.dropna(subset=['total_sqft'])

# 4. price: convert to float
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df[df['price'].notnull()]  # Drop rows with missing price


# Drop unrealistic entries
df = df[df['BHK'] > 0]
df = df[df['total_sqft'] / df['BHK'] >= 300]


# Reset index
df.reset_index(drop=True, inplace=True)


# Save Cleaned Dataset
df.to_csv("cleaned_india_real_estate.csv", index=False)
print("Cleaned data saved as 'cleaned_india_real_estate.csv'")
print("Final shape:", df.shape)