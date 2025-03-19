import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------
# 🎯 Load the Trained Model
# -------------------------------------------
import os
import joblib

# Correct model path for Streamlit Cloud
model_path = os.path.join(os.path.dirname(__file__), "models", "xgboost_finetuned_model.pkl")
model = joblib.load(model_path)

# -------------------------------------------
# 🎨 Streamlit UI Configuration
# -------------------------------------------
st.title("🏗️ Concrete Strength Prediction UI")
st.write(
    "Enter the concrete mix design parameters below to predict the compressive strength."
)

# -------------------------------------------
# 📥 Define Input Fields for User
# -------------------------------------------
cement = st.number_input("Cement (kg/m³)", min_value=50.0, max_value=600.0, value=150.0)
slag = st.number_input(
    "Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=300.0, value=0.0
)
fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=200.0, value=0.0)
water = st.number_input("Water (kg/m³)", min_value=100.0, max_value=250.0, value=150.0)
superplasticizer = st.number_input(
    "Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0, value=0.0
)
coarse_agg = st.number_input(
    "Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1200.0, value=1000.0
)
fine_agg = st.number_input(
    "Fine Aggregate (kg/m³)", min_value=500.0, max_value=1000.0, value=800.0
)
age = st.number_input("Age (days)", min_value=1, max_value=365, value=28)

# -------------------------------------------
# 📊 Collect Input Data into Dictionary
# -------------------------------------------
input_data = {
    "Cement": cement,
    "Blast Furnace Slag": slag,
    "Fly Ash": fly_ash,
    "Water": water,
    "Superplasticizer": superplasticizer,
    "Coarse Aggregate": coarse_agg,
    "Fine Aggregate": fine_agg,
    "Age": age,
}

# -------------------------------------------
# 🧠 Add Feature Engineering for Derived Ratios
# -------------------------------------------
input_data["Water_to_Cement"] = input_data["Water"] / input_data["Cement"]
input_data["Water_to_Binder"] = input_data["Water"] / (
    input_data["Cement"] + input_data["Fly Ash"] + input_data["Blast Furnace Slag"]
)
input_data["Superplasticizer_to_Binder"] = input_data["Superplasticizer"] / (
    input_data["Cement"] + input_data["Fly Ash"] + input_data["Blast Furnace Slag"]
)
input_data["Fly_Ash_to_Cement"] = input_data["Fly Ash"] / input_data["Cement"]
input_data["Slag_to_Binder"] = input_data["Blast Furnace Slag"] / (
    input_data["Cement"] + input_data["Fly Ash"] + input_data["Blast Furnace Slag"]
)

# -------------------------------------------
# 📊 Convert Input to DataFrame
# -------------------------------------------
input_df = pd.DataFrame([input_data])

# -------------------------------------------
# 🎯 Validate Input Based on Augmentation Constraints
# -------------------------------------------
def validate_input(data):
    ranges = {
        "Cement": (50, 600),
        "Blast Furnace Slag": (0, 300),
        "Fly Ash": (0, 200),
        "Water": (100, 250),
        "Superplasticizer": (0, 50),
        "Coarse Aggregate": (800, 1200),
        "Fine Aggregate": (500, 1000),
        "Age": (1, 365),
    }

    for col, (min_val, max_val) in ranges.items():
        if not (min_val <= data[col] <= max_val):
            st.error(f"❌ {col} is out of bounds! Must be between {min_val} and {max_val}.")
            return False
    return True


# -------------------------------------------
# 🚀 Predict Button and Model Prediction
# -------------------------------------------
if st.button("🔮 Predict Strength"):
    if validate_input(input_data):
        try:
            # Make Prediction
            predicted_strength = model.predict(input_df)[0]
            st.success(f"✅ Predicted Concrete Strength: **{predicted_strength:.2f} MPa**")

        except Exception as e:
            st.error(f"❌ Error occurred while predicting: {e}")

# -------------------------------------------
# 📚 Display Limitations of the Model
# -------------------------------------------
st.write("### ⚠️ Model Limitations")
st.info(
    """
- Model trained on data with specific material limits. Values beyond the limits may reduce prediction accuracy.
- Synthetic data created using Monte Carlo and Gaussian noise may not account for all possible mix designs.
- Augmentation approximates real-world conditions but cannot fully replace physical testing.
"""
)

