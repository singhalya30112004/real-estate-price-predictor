import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load("real_estate_price_model.pkl")

# Extract components
preprocessor = model.named_steps['preprocessing']
regressor = model.named_steps['model']

# Load original data
df = pd.read_csv("cleaned_india_real_estate.csv")

# Add missing feature used in training
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Subset training features
X_sample = df[['BHK', 'bathroom', 'total_sqft', 'city', 'location', 'locality_type', 'price_per_sqft']]

# Use the pipeline itself to transform
X_transformed = preprocessor.transform(X_sample)

# Get correct feature names
feature_names = preprocessor.get_feature_names_out()

# Final length check
assert X_transformed.shape[1] == len(feature_names) == len(regressor.feature_importances_), \
    f"Shape mismatch: {X_transformed.shape[1]} transformed features, {len(feature_names)} names, {len(regressor.feature_importances_)} importances"

# Build DataFrame for importance
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Clean feature names
def clean_label(label):
    if label.startswith("remainder__"):
        return label.replace("remainder__", "").replace("_", " ").title()
    elif label.startswith("cat__city_"):
        return "City: " + label.replace("cat__city_", "").replace("_", " ").title()
    elif label.startswith("cat__location_"):
        return "Location: " + label.replace("cat__location_", "").replace("_", " ").title()
    elif label.startswith("cat__locality_type_"):
        return "Locality Type: " + label.replace("cat__locality_type_", "").replace("_", " ").title()
    else:
        return label.replace("_", " ").title()

feat_df["Clean Feature"] = feat_df["Feature"].apply(clean_label)

# Plot top 15
plt.figure(figsize=(10, 8))
top_n = 15
plt.barh(feat_df['Clean Feature'][:top_n][::-1], feat_df['Importance'][:top_n][::-1], color='slateblue')
plt.xlabel("Importance Score (Log Scale)")
plt.title(f"Top {top_n} Important Features for House Price Prediction")
plt.xscale('log')  # Makes smaller bars more visible
plt.tight_layout()
plt.savefig("feature_importance_logscale_cleaned.png")
plt.show()