# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")

# ============================
# 1. Load Dataset (hardcoded)
# ============================
DATA_PATH = "AI_Invasion_In-Class_Dataset.xlsx"
df = pd.read_excel(DATA_PATH)

st.subheader("Dataset Preview")
st.write(df.head())

# ============================
# 2. Data Cleaning
# ============================
st.subheader("Data Cleaning")
st.write("Missing Values Before:")
st.write(df.isnull().sum())

mean_value = df["Distance_Km"].mean()
df["Distance_Km"].fillna(mean_value, inplace=True)

st.write("Missing Values After:")
st.write(df.isnull().sum())

# Drop 'Model' column if it exists
if "Model" in df.columns:
    df.drop("Model", axis=1, inplace=True)

# Encode categorical features
cat_features = ["Location", "Maker", "Year", "Colour", "Type"]
for cat_feature in cat_features:
    df[f"{cat_feature}_cat"] = df[cat_feature].astype('category').cat.codes

df.drop(cat_features, axis=1, inplace=True)

# ============================
# 3. Features & Target
# ============================
y = df["Amount (Million ₦)"]
X = df.drop("Amount (Million ₦)", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================
# 4. Train models
# ============================
st.subheader("Model Training & Evaluation")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regressor": SVR()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = mae

st.write("📊 Mean Absolute Error (MAE) for each model:")
st.json(results)

best_model_name = min(results, key=results.get)
st.success(f"✅ Best Model: {best_model_name} with MAE = {results[best_model_name]:.2f}")

# ============================
# 5. Interactive Prediction
# ============================
st.subheader("Make a Prediction")
distance = st.number_input("Enter Distance (Km)", min_value=0, step=1000)
location = st.text_input("Location (e.g., Abuja, Lagos)")
maker = st.text_input("Maker (e.g., Toyota, Lexus)")
year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
colour = st.text_input("Colour (e.g., Black, White)")
car_type = st.text_input("Type (e.g., Foreign Used, Nigerian Used, Brand New)")

if st.button("Predict Price"):
    # Encode inputs like training
    input_data = pd.DataFrame([{
        "Distance_Km": distance,
        "Location_cat": pd.Series([location]).astype("category").cat.codes[0],
        "Maker_cat": pd.Series([maker]).astype("category").cat.codes[0],
        "Year_cat": pd.Series([year]).astype("category").cat.codes[0],
        "Colour_cat": pd.Series([colour]).astype("category").cat.codes[0],
        "Type_cat": pd.Series([car_type]).astype("category").cat.codes[0]
    }])

    best_model = models[best_model_name]
    prediction = best_model.predict(input_data)[0]
    st.success(f"💰 Predicted Price: {prediction:.2f} Million ₦")
