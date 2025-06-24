#!/opt/anaconda3/bin/python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load cleaned dataset
df = pd.read_csv("cleaned_india_real_estate.csv")

# Calculate price per sqft (important feature!)
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Define features & target
X = df[['BHK', 'bathroom', 'total_sqft', 'price_per_sqft', 'city', 'location', 'locality_type']]
y = df['price']


# Preprocessing: One-hot encode categorical vars
categorical_cols = ['city', 'location', 'locality_type']
numeric_cols = ['BHK', 'bathroom', 'total_sqft', 'price_per_sqft']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep numeric columns
)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = model_pipeline.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ₹{rmse:,.0f}")