import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.regression import *
import os


os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

st.set_page_config(page_title="Manufacturing Defect Optimization Dashboard", layout="wide")
st.title("🏭 Manufacturing Process Optimization Dashboard")
st.subheader("📂 Data Source")


uploaded_file = st.file_uploader("Upload Manufacturing Dataset (CSV)", type=["csv"])


def generator_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "machine_speed": np.random.uniform(50, 150, 500),
        "temperature": np.random.uniform(250, 400, 500),
        "pressure": np.random.uniform(2, 10, 500),
        "material_quality": np.random.randint(1, 10, 500),
        "operator_experience": np.random.randint(1, 15, 500)
    })
    data["defect_rate"] = (
        0.04 * data["machine_speed"]
        - 0.03 * data["temperature"]
        - 0.5 * data["material_quality"]
        - 0.3 * data["operator_experience"]
        + np.random.normal(0, 4, 500)
    )
    return data


if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.success("✅ Real manufacturing dataset loaded")
else:
    raw_data = generator_data()
    st.info("ℹ️ Using synthetic demo dataset")


raw_data.columns = raw_data.columns.str.strip()


raw_data.to_csv("data/raw/raw_data.csv", index=False)


st.subheader("📋 Dataset Preview")
st.dataframe(raw_data.head())
st.write("📌 Available Columns:", raw_data.columns.tolist())


st.subheader("🎯 Select Target Column")
target_col = st.selectbox(
    "Choose the column you want to predict",
    raw_data.columns
)
st.success(f"Selected Target Column: **{target_col}**")


st.subheader("🔍 Data Quality Checks")
missing_values = raw_data.isnull().sum().sum()
st.metric("Missing Values", missing_values)
z_scores = np.abs((raw_data[target_col] - raw_data[target_col].mean()) / raw_data[target_col].std())
outliers = (z_scores > 3).sum()
st.metric("Outliers Detected", outliers)

st.subheader("🛠 Feature Engineering")
data = raw_data.copy()


if "machine_speed" in data.columns and "temperature" in data.columns:
    data["speed_temp_ratio"] = data["machine_speed"] / data["temperature"]

if "operator_experience" in data.columns:
    data["experience_level"] = pd.cut(
        data["operator_experience"],
        bins=[0, 5, 10, 20],
        labels=["Junior", "Mid", "Senior"]
    )

st.success("✅ Feature engineering completed where possible")
data.to_csv("data/processed/processed_data.csv", index=False)


numeric_features = data.select_dtypes(include=np.number).columns.tolist()
if target_col in numeric_features:
    numeric_features.remove(target_col)  # Remove target from features

st.subheader("🤖 Model Training & Selection")
with st.spinner("⏳ Training multiple models..."):
    setup(
        data=data[numeric_features + [target_col]],
        target=target_col,
        session_id=123,
        normalize=True
    )
    best_model = compare_models()
    final_model = finalize_model(best_model)

st.success("✅ Model training completed")
st.markdown(f"**Selected Best Model:** `{best_model.__class__.__name__}`")


st.subheader("🧮 What-If Analysis (User Input)")
input_data = pd.DataFrame(columns=numeric_features)


for col in numeric_features:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    mean_val = float(data[col].mean())
    input_data[col] = [st.slider(f"{col}", min_val, max_val, mean_val)]


prediction = predict_model(final_model, data=input_data)
predicted_value = prediction["prediction_label"][0]


st.subheader("📈 Prediction Result")
st.metric(f"Predicted {target_col}", f"{predicted_value:.2f}")


if predicted_value < 2:
    st.success("🟢 Low Defect Risk")
elif predicted_value < 5:
    st.warning("🟡 Medium Defect Risk")
else:
    st.error("🔴 High Defect Risk")
st.subheader("📊 Feature Importance (SHAP)")

try:
    plot_model(final_model, plot="shap", display_format="streamlit")
except:
    st.info("SHAP explanation not availablee")



st.subheader(f"📉 {target_col} Distribution")
fig, ax = plt.subplots()
ax.hist(data[target_col], bins=30)
ax.axvline(predicted_value, color='red', linestyle='--', label='Prediction')
ax.set_xlabel(target_col)
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)
