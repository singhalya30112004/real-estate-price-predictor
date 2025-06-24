# Real Estate Price Predictor

A machine learning project that predicts house prices in Indian cities (Delhi, Bangalore) based on features like location, BHK, square footage, and more. Trained on real datasets and optimised for accuracy, this project demonstrates practical data science skills from data cleaning to model deployment.


## Dataset

Combined and cleaned real-world datasets from:
- **Bangalore**
- **Delhi**

Columns used include: `city`, `location`, `locality_type`, `BHK`, `bathroom`, `total_sqft`, `furnishing`, `price`, `parking` (dropped), `balcony` (dropped)


## Workflow

### 1. Data Cleaning (`clean_real_estate_data.py`)
- Handled missing values
- Converted sqft ranges (e.g., "1000–1200" → average)
- Dropped outliers (e.g., 100 BHK homes)

### 2. Exploratory Data Analysis (`eda_real_estate.py`)
- Price distribution
- City-wise comparison
- Correlation heatmap
- BHK vs Price boxplots

### 3. Modeling (`model_train_real_estate.py`)
- Feature preprocessing with One-Hot Encoding
- Trained multiple models:
  - Linear Regression
  - Random Forest Regressor
- **Best model:** `RandomForestRegressor`
  - R² Score: **0.992**
  - RMSE: ₹ **13.47 Lakhs**

### 4. Feature Importance (`feature_importance_real_estate.py`)
- Visualized which features matter most for price prediction

### 5. Predictions (`predict_real_estate.py`)
- Predicts prices for new properties using the saved model pipeline


## Visuals

### Price Distribution (Zoomed In)
<img width="1381" alt="image" src="https://github.com/user-attachments/assets/260e8c49-8332-4981-b9e3-1c2666cc5dac" />

### Price Per Sqft Distribution
<img width="1381" alt="image" src="https://github.com/user-attachments/assets/0d550b1b-9285-46e3-8734-595d73c32cc1" />

### Area vs Price
<img width="1394" alt="image" src="https://github.com/user-attachments/assets/9518effd-04ae-468e-9850-c3788e234624" />


## Tech Stack

- `Python`
- `Pandas`, `NumPy` for data handling
- `Matplotlib`, `Seaborn` for visualisation
- `Scikit-Learn`, `XGBoost` for modeling
- `Joblib` for saving model and pipeline


##  How to Run

```bash
# Clone the repo
git clone https://github.com/singhalya30112004/real-estate-price-predictor
cd real-estate-price-predictor

# Install requirements
pip install -r requirements.txt

# Step-by-step execution
python clean_real_estate_data.py
python eda_real_estate.py
python model_train_real_estate.py
python feature_importance_real_estate.py
python predict_real_estate.py
```


## Sample Prediction

```python
sample = {
    'BHK': 3,
    'bathroom': 2,
    'total_sqft': 1500,
    'city': 'bangalore',
    'location': 'indiranagar',
    'locality_type': 'residential'
}

# Output: Predicted Price ≈ ₹1.12 Cr
```

## Author

Alya Singh  
[LinkedIn](https://www.linkedin.com/in/alya-singh/)  
[GitHub: @singhalya30112004](https://github.com/singhalya30112004)


## License
MIT License – feel free to use, share, and modify!
