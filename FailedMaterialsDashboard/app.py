import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Material Failure Dashboard", layout="wide")
st.title("🏗️ Construction Material Failure Prediction Dashboard")

st.write(
    "This dashboard demonstrates how **machine learning** can support "
    "early-stage **construction material quality assessment** using synthetic data."
)

# -----------------------------------
# DATA GENERATION
# -----------------------------------
st.header("📊 Synthetic Data Generation")

n_samples = st.slider("Number of material samples", 500, 5000, 1000)

np.random.seed(42)

data = pd.DataFrame({
    "cement_ratio": np.random.uniform(0.2, 0.6, n_samples),
    "aggregate_ratio": np.random.uniform(0.3, 0.7, n_samples),
    "water_ratio": np.random.uniform(0.1, 0.4, n_samples),
    "curing_temperature": np.random.uniform(20, 90, n_samples),
    "curing_time_days": np.random.uniform(3, 60, n_samples),
    "manufacturing_pressure": np.random.uniform(100, 500, n_samples)
})

# -----------------------------------
# STRESS SCORE (SYNTHETIC LOGIC)
# -----------------------------------
stress_score = (
    0.45 * data["cement_ratio"]
    - 0.35 * data["water_ratio"]
    + 0.20 * (data["curing_time_days"] / 60)
    + 0.25 * (data["curing_temperature"] / 90)
    + 0.30 * (data["manufacturing_pressure"] / 500)
)

# add noise for realism
noise = np.random.normal(0, 0.05, n_samples)
stress_score = stress_score + noise

data["pass_fail"] = (stress_score > 0.55).astype(int)

# -----------------------------------
# KPI SECTION
# -----------------------------------
st.header("📌 Key Performance Indicators")

total = len(data)
pass_count = data["pass_fail"].sum()
fail_count = total - pass_count
pass_rate = (pass_count / total) * 100
fail_rate = 100 - pass_rate

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Samples", total)
col2.metric("Pass Rate (%)", f"{pass_rate:.1f}")
col3.metric("Fail Rate (%)", f"{fail_rate:.1f}")

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
X = data.drop("pass_fail", axis=1)
y = data["pass_fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    accuracies[name] = accuracy_score(y_test, preds) * 100

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

col4.metric("Best Model", best_model_name)

# -----------------------------------
# MODEL COMPARISON CHART
# -----------------------------------
st.header("📈 Model Performance Comparison")

fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison")
st.pyplot(fig)

# -----------------------------------
# DATA DISTRIBUTION
# -----------------------------------
st.header("📊 Data Distribution")

fig2, ax2 = plt.subplots()
data["pass_fail"].value_counts().plot(kind="bar", ax=ax2)
ax2.set_xticklabels(["FAIL", "PASS"], rotation=0)
ax2.set_ylabel("Count")
ax2.set_title("Pass vs Fail Distribution")
st.pyplot(fig2)

# -----------------------------------
# PREDICTION SECTION
# -----------------------------------
st.header("🔍 Material Failure Prediction")

with st.form("prediction_form"):
    cement = st.number_input("Cement Ratio", 0.2, 0.6, 0.4)
    aggregate = st.number_input("Aggregate Ratio", 0.3, 0.7, 0.5)
    water = st.number_input("Water Ratio", 0.1, 0.4, 0.25)
    temp = st.number_input("Curing Temperature (°C)", 20, 90, 50)
    days = st.number_input("Curing Time (Days)", 3, 60, 28)
    pressure = st.number_input("Manufacturing Pressure", 100, 500, 300)

    submit = st.form_submit_button("Predict")

if submit:
    user_data = np.array([[cement, aggregate, water, temp, days, pressure]])
    user_scaled = scaler.transform(user_data)

    prediction = best_model.predict(user_scaled)[0]
    probabilities = best_model.predict_proba(user_scaled)[0]

    fail_prob = probabilities[0] * 100
    pass_prob = probabilities[1] * 100
    confidence = max(fail_prob, pass_prob)

    # Risk level
    if confidence < 60:
        risk = "🟢 Low Risk"
    elif confidence < 80:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    if prediction == 1:
        st.success(f"✅ PASSES (Confidence: {pass_prob:.1f}%)")
    else:
        st.error(f"❌ FAILS (Confidence: {fail_prob:.1f}%)")

    st.write(f"**Risk Level:** {risk}")

    # WHY explanation
    st.subheader("🧠 Explanation (Why this result?)")

    if cement > 0.45:
        st.write("✔ High cement ratio improves strength")
    else:
        st.write("✖ Low cement ratio reduces strength")

    if water < 0.25:
        st.write("✔ Low water ratio improves durability")
    else:
        st.write("✖ High water ratio weakens material")

    if days > 28:
        st.write("✔ Longer curing time improves bonding")
    else:
        st.write("✖ Short curing time reduces strength")

    if temp > 50:
        st.write("✔ Higher curing temperature improves curing")
    else:
        st.write("✖ Low curing temperature slows curing")

    if pressure > 300:
        st.write("✔ High manufacturing pressure increases density")
    else:
        st.write("✖ Low pressure reduces compaction")

# -----------------------------------
# LIMITATIONS
# -----------------------------------
st.markdown("---")
st.header("🔄 What-If Analysis (Parameter Sensitivity)")

st.write(
    "This analysis shows how changing **one parameter** affects "
    "material performance while keeping others constant."
)

# Select parameter
parameter = st.selectbox(
    "Select parameter to vary",
    [
        "cement_ratio",
        "water_ratio",
        "curing_temperature",
        "curing_time_days",
        "manufacturing_pressure"
    ]
)

# Slider range based on parameter
param_ranges = {
    "cement_ratio": (0.2, 0.6),
    "water_ratio": (0.1, 0.4),
    "curing_temperature": (20, 90),
    "curing_time_days": (3, 60),
    "manufacturing_pressure": (100, 500)
}

new_value = st.slider(
    f"Change {parameter}",
    param_ranges[parameter][0],
    param_ranges[parameter][1],
    float(locals()[parameter.split('_')[0]]) if parameter != "curing_time_days" else float(days)
)

# Copy original input
what_if_data = {
    "cement_ratio": cement,
    "aggregate_ratio": aggregate,
    "water_ratio": water,
    "curing_temperature": temp,
    "curing_time_days": days,
    "manufacturing_pressure": pressure
}

# Update selected parameter
what_if_data[parameter] = new_value

# Predict again
what_if_array = np.array([list(what_if_data.values())])
what_if_scaled = scaler.transform(what_if_array)
what_if_pred = best_model.predict(what_if_scaled)[0]
what_if_prob = best_model.predict_proba(what_if_scaled)[0]

st.subheader("📌 What-If Result")

if what_if_pred == 1:
    st.success(f"✅ PASSES (Confidence: {what_if_prob[1]*100:.1f}%)")
else:
    st.error(f"❌ FAILS (Confidence: {what_if_prob[0]*100:.1f}%)")

st.markdown("---")
st.subheader("⚠️ Project Limitations")
st.write(
    "- This dashboard uses **synthetic data** for educational purposes.\n"
    "- Predictions should be treated as **decision support**, not final judgments.\n"
    "- Real-world material behavior depends on additional chemical and environmental factors."
)
