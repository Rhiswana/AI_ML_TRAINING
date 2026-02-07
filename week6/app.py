import streamlit as st
import pandas as pd
import numpy as np
from pycaret.anomaly import load_model, predict_model
from datetime import datetime

st.set_page_config(page_title="Factory Anomaly Detection", layout="wide")

if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0
if "show_alert" not in st.session_state:
    st.session_state.show_alert = False
if "risk_message" not in st.session_state:
    st.session_state.risk_message = None
if "risk_type" not in st.session_state:
    st.session_state.risk_type = None

st.title("🏭 Factory Anomaly Detection Dashboard")
st.write("Simulated real-time monitoring of machine sensor health")

DOWNTIME_COST_PER_HR = 5000

NORMAL_RANGES = {
    "temperature": (55, 65),
    "pressure": (25, 35),
    "vibration": (8, 12)
}

model = load_model("factory_anomaly_model")

st.subheader("📊 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Anomalies Today", st.session_state.anomaly_count)
col2.metric("System Uptime", "98.5%")
col3.metric("Last Check", datetime.now().strftime("%H:%M:%S"))

if st.session_state.risk_message:
    if st.session_state.risk_type == "success":
        st.success(st.session_state.risk_message)
    elif st.session_state.risk_type == "warning":
        st.warning(st.session_state.risk_message)
    else:
        st.error(st.session_state.risk_message)

if st.session_state.show_alert:
    st.subheader("💰 Downtime Cost Estimate")
    st.write(f"Estimated loss if ignored: **${DOWNTIME_COST_PER_HR:,} per hour**")
    st.session_state.show_alert = False

st.subheader("🔧 Enter Live Sensor Readings")
c1, c2, c3 = st.columns(3)
temperature = c1.number_input("Temperature (°C)", value=60.0)
pressure = c2.number_input("Pressure (bar)", value=30.0)
vibration = c3.number_input("Vibration (Hz)", value=10.0)

st.subheader("📏 Normal Operating Ranges")
st.write(f"**Temperature:** 55–65 °C | Your Reading: **{temperature} °C**")
st.write(f"**Pressure:** 25–35 bar | Your Reading: **{pressure} bar**")
st.write(f"**Vibration:** 8–12 Hz | Your Reading: **{vibration} Hz**")

if st.button("🚨 Check Machine Health"):
    input_data = pd.DataFrame({
        "temperature": [temperature],
        "pressure": [pressure],
        "vibration": [vibration]
    })

    result = predict_model(model, input_data)
    anomaly = result["Anomaly"][0]

    risk_score = 0
    if temperature > NORMAL_RANGES["temperature"][1]:
        risk_score += 1
    if pressure > NORMAL_RANGES["pressure"][1]:
        risk_score += 1
    if vibration > NORMAL_RANGES["vibration"][1]:
        risk_score += 1

    if risk_score == 0:
        st.session_state.risk_message = "🟢 Low Risk: Machine operating normally"
        st.session_state.risk_type = "success"
    elif risk_score == 1:
        st.session_state.risk_message = "🟠 Medium Risk: Monitor machine closely"
        st.session_state.risk_type = "warning"
    else:
        st.session_state.risk_message = "🔴 Critical Risk: Immediate action required"
        st.session_state.risk_type = "error"

    if anomaly == 1:
        st.session_state.anomaly_count += 1
        st.session_state.show_alert = True

    st.rerun()

st.subheader("📈 Sensor Trend Analysis (Simulated)")
trend_data = pd.DataFrame({
    "temperature": np.random.normal(60, 5, 50),
    "pressure": np.random.normal(30, 3, 50),
    "vibration": np.random.normal(10, 1, 50)
})

sensor_choice = st.selectbox(
    "Select Sensor to View Trend",
    ["temperature", "pressure", "vibration"]
)

low, high = NORMAL_RANGES[sensor_choice]
st.write(f"**Normal Range:** {low} – {high}")

latest_value = trend_data[sensor_choice].iloc[-1]
st.metric("Latest Reading", round(latest_value, 2))

st.line_chart(trend_data[sensor_choice].tail(20))
