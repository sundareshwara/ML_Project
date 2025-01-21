import streamlit as st
import joblib
import gdown

# Google Drive link
model_url = 'https://drive.google.com/uc?export=download&id=102TmWw29JeeV0onEIZIDSK0a0LQwm6Fq'
model_path = '100k_trained_model.pkl'

# Download the model file from Google Drive
gdown.download(model_url, model_path, quiet=False)

# Load the trained model
model = joblib.load(model_path)

# Your app code continues here...
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("Concrete Strength Predictor")
st.write("🔍 Use this app to predict the compressive strength of concrete based on input parameters.")

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("""
- 📊 **Predict Concrete Strength**
- ℹ️ **About**
""")

# Create tabs for better organization
tab1, tab2 = st.tabs(["🧪 Predict", "📘 About"])

# Tab 1: Prediction
with tab1:
    st.header("Input Parameters")
    st.markdown("Enter the values below to predict the compressive strength of concrete:")
    
    # Input text boxes for user inputs
    cement = st.text_input("Cement (kg/m³):", "300")
    slag = st.text_input("Blast Furnace Slag (kg/m³):", "100")
    fly_ash = st.text_input("Fly Ash (kg/m³):", "50")
    water = st.text_input("Water (kg/m³):", "200")
    superplasticizer = st.text_input("Superplasticizer (kg/m³):", "5")
    coarse_aggregate = st.text_input("Coarse Aggregate (kg/m³):", "900")
    fine_aggregate = st.text_input("Fine Aggregate (kg/m³):", "700")
    age = st.text_input("Age (days):", "28")

    # Predict button
    if st.button("🔮 Predict"):
        try:
            # Convert inputs to float for prediction
            new_data = [[
                float(cement),
                float(slag),
                float(fly_ash),
                float(water),
                float(superplasticizer),
                float(coarse_aggregate),
                float(fine_aggregate),
                float(age)
            ]]
            
            # Make prediction
            prediction = model.predict(new_data)
            
            # Display the result
            st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")
        except ValueError:
            st.error("Please ensure all inputs are numeric.")

# Tab 2: About the App
with tab2:
    st.header("About")
    st.markdown("""
    - **Purpose**: This app predicts the compressive strength of concrete based on its mix proportions and age.
    - **Model Used**: Random Forest Regressor trained on a dataset of concrete mix properties.
    - **Developer**: [Your Name](https://your-profile-link.com)
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Concrete_in_slab.jpg", caption="Concrete Slab Example")

    st.markdown("""
    **Disclaimer**: The predictions provided are for educational purposes only. For precise results, consult with industry experts.
    """)
