import pandas as pd
import joblib

# Load pipeline
model = joblib.load("real_estate_price_model.pkl")

# New data
data = pd.DataFrame([
    {
        "BHK": 3,
        "bathroom": 2,
        "total_sqft": 1400,
        "price_per_sqft": 8500,
        "city": "bangalore",
        "location": "whitefield",
        "locality_type": "residential area"
    }
    ,
    {
        "BHK": 2,
        "bathroom": 2,
        "total_sqft": 950,
        "price_per_sqft": 11000,
        "city": "delhi",
        "location": "dwarka",
        "locality_type": "apartment"
    }
])

# Predict
predictions = model.predict(data)

# Show output
for i, price in enumerate(predictions):
    print(f"\nPrediction {i+1}")
    print(f"Estimated Price: ₹{price:,.0f} = ₹ {price / 1e7:.2f} Crores")
