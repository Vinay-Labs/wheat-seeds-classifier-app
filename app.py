import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


# It should at the top, right after imports.
st.set_page_config(page_title="Wheat Seed Classifier", page_icon="🌾")


# 1. Load Assets
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('seeds_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None


model, scaler = load_assets()

# 2. App Title
st.title("🌾 Wheat Seed Classifier")

if model is None or scaler is None:
    st.error(
        "Error: Could not load 'seeds_model.keras' or 'scaler.pkl'. Please make sure they are in the same folder as app.py.")
    st.stop()

st.markdown(
    "Enter the geometric measurements of the seed to predict its variety (**Kama**, **Rosa**, or **Canadian**).")

# Sidebar Guide
st.sidebar.header("ℹ️ Typical Data Ranges")
st.sidebar.markdown("""
Based on dataset analysis:
* **Area:** 10.5 - 21.2
* **Perimeter:** 12.4 - 17.3
* **Length:** 4.9 - 6.7
* **Width:** 2.6 - 4.1
""")

# 3. User Inputs
st.subheader("Seed Measurements")
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", 10.0, 22.0, 14.85)
    perimeter = st.number_input("Perimeter", 12.0, 18.0, 14.56)
    compactness = st.number_input("Compactness", 0.800, 0.950, 0.871, step=0.001, format="%.4f")
    len_kernel = st.number_input("Length of Kernel", 4.50, 7.00, 5.63)

with col2:
    width_kernel = st.number_input("Width of Kernel", 2.50, 4.50, 3.26)
    asymmetry = st.number_input("Asymmetry Coeff", 0.50, 9.00, 3.70)
    len_groove = st.number_input("Length of Groove", 4.00, 7.00, 5.41)

# 4. DATA VALIDATION
valid = True

if width_kernel >= len_kernel:
    st.error("⚠️ **Logical Error:** Width cannot be greater than Length.")
    valid = False

if len_groove > len_kernel:
    st.error("⚠️ **Logical Error:** Groove length cannot exceed Kernel length.")
    valid = False

calc_compactness = (4 * np.pi * area) / (perimeter ** 2)
if abs(compactness - calc_compactness) > 0.05:
    st.warning(
        f"💡 **Note:** Based on Area and Perimeter, Compactness should be approx **{calc_compactness:.3f}**. You entered **{compactness}**.")

# 5. Prediction
if st.button("Predict Variety"):
    if valid:
        user_input = np.array([[area, perimeter, compactness, len_kernel, width_kernel, asymmetry, len_groove]])
        user_input_scaled = scaler.transform(user_input)

        probs = model.predict(user_input_scaled)
        pred_class = np.argmax(probs)
        confidence = np.max(probs) * 100

        classes = ['Kama', 'Rosa', 'Canadian']
        result = classes[pred_class]

        st.success(f"### Prediction: {result}")
        st.info(f"Model Confidence: {confidence:.2f}%")

        st.write("Probability Breakdown:")
        df_probs = pd.DataFrame(probs[0], index=classes, columns=['Probability'])
        st.bar_chart(df_probs)

    else:
        st.error("Please fix errors before predicting.")