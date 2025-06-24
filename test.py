import joblib

# Load model
model = joblib.load("real_estate_price_model.pkl")

# Extract pipeline components
preprocessor = model.named_steps['preprocessing']
regressor = model.named_steps['model']

# Get numeric features (from ColumnTransformer)
numeric_features = preprocessor.transformers_[0][2]  # Already a list

# Get encoded feature names from OneHotEncoder
encoded_cat_features = preprocessor.transformers_[1][1].get_feature_names_out()

# Merge
all_features = list(numeric_features) + list(encoded_cat_features)

# Get importances
importances = regressor.feature_importances_

# Debug lengths
print("🔍 numeric_features:", len(numeric_features))
print("🔍 encoded_cat_features:", len(encoded_cat_features))
print("🔍 Total features:", len(all_features))
print("🧠 Importances returned by model:", len(importances))
