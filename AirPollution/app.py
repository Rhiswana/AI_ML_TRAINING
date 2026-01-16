import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("india_air_pollution.csv")
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "severe"

st.title("🌆 Indian Air Pollution Dashboard")
st.markdown("""
This app visualizes air pollution levels across major Indian cities.
You can explore AQI, PM2.5, PM10, and temperature.
""")


st.sidebar.header("User control")
selected_cities=st.sidebar.multiselect("select city",
                                       options=data["City"].unique(),
                                       default=data["City"].unique()
                                       )
filtered_data=data[data["City"].isin(selected_cities)]

chart_type = st.sidebar.selectbox("Select chart type", ["Bar Chart", "Pie Chart", "Scatter Plot"])
column = st.sidebar.selectbox("Select column to visualize", ["AQI", "PM2.5", "PM10", "Temperature", "Humidity"])
color = st.sidebar.color_picker("Pick a color for the chart", "#1f77b4")
aqi_threshold=st.sidebar.number_input("set AQI Health threshold", min_value=0, max_value=500, value=200)
if filtered_data.empty:
    st.warning("Please select at least one city.")
    st.stop()
filtered_data["AQI Category"] = filtered_data["AQI"].apply(aqi_category)
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average AQI", round(filtered_data["AQI"].mean(), 1))
col2.metric("Maximum AQI", filtered_data["AQI"].max())
col3.metric("Minimum AQI", filtered_data["AQI"].min())

if chart_type == "Bar Chart":


    st.subheader(f" {column} Comparsion across Cities")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="City", y=column, data=filtered_data, color=color, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif chart_type == "Pie Chart":
    st.subheader(f" {column} distribution by city")
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(filtered_data[column], labels=filtered_data["City"], autopct="%1.1f%%")
    st.pyplot(fig)

elif chart_type == "Scatter Plot":
    st.subheader(f"Relationship Between Temperature and {column}")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x="Temperature", y=column, hue="City",  s=100, data=filtered_data, ax=ax)
    st.pyplot(fig)
    st.subheader("📌 Health Insights")
    high_pollution=filtered_data[filtered_data["AQI"]>aqi_threshold]
    if not filtered_data.empty:
        st.warning(
            f"⚠️{len(high_pollution)}city(s) exceed the AQI threshold of {aqi_threshold}."
            "These cities may pose serious health risks."
        )
        st.dataframe(high_pollution[["City","AQI"]])
    else:
        st.success("✅ All selected cities are within the safe AQI limit.")
worst_city = filtered_data.loc[filtered_data["AQI"].idxmax()]
best_city = filtered_data.loc[filtered_data["AQI"].idxmin()]

st.info(f"🚨 Worst Air Quality: {worst_city['City']} (AQI: {worst_city['AQI']})")
st.success(f"🌿 Best Air Quality: {best_city['City']} (AQI: {best_city['AQI']})")

st.subheader("➕ Analyze User-Entered City Data")

city_name = st.text_input("City Name")
user_aqi = st.number_input("AQI", min_value=0, max_value=500)
user_pm25 = st.number_input("PM2.5", min_value=0)
user_pm10 = st.number_input("PM10", min_value=0)
user_temp = st.number_input("Temperature (°C)")
user_humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)

user_df = None

if st.button("Analyze City"):
    if city_name.strip() == "":
        st.error("Please enter a city name.")
    else:
        category = aqi_category(user_aqi)

        user_df = pd.DataFrame([{
            "City": city_name,
            "AQI": user_aqi,
            "PM2.5": user_pm25,
            "PM10": user_pm10,
            "Temperature": user_temp,
            "Humidity": user_humidity,
            "AQI Category": category
        }])

        st.subheader("User City Air Quality Analysis")
        st.dataframe(user_df)
        plot_data = pd.concat([filtered_data, user_df], ignore_index=True)

        if category in ["Poor", "Very Poor", "Severe"]:
            st.error("⚠️ Health Risk: Avoid outdoor activities.")
        else:
            st.success("✅ Air quality is acceptable.")
        st.subheader("📊 Visual Comparison")

        col1, col2 = st.columns(2)

        # BAR CHART
        with col1:
            st.markdown("**AQI Comparison (Bar Chart)**")
            fig, ax = plt.subplots()
            sns.barplot(x="City", y="AQI", data=plot_data, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with col2:
            st.markdown("**AQI Distribution (Pie Chart)**")
            fig, ax = plt.subplots()
            ax.pie(
                plot_data["AQI"],
                labels=plot_data["City"],
                autopct="%1.1f%%"
            )
            st.pyplot(fig)
        st.markdown("**Temperature vs AQI (Scatter Plot)**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x="Temperature",
            y="AQI",
            hue="City",
            data=plot_data,
            s=120,
            ax=ax
        )
        sns.scatterplot(
            x=[user_temp],
            y=[user_aqi],
            s=300,
            marker="X",
            color="red",
            label="User City",
            ax=ax
        )

        st.pyplot(fig)
    st.download_button(
        "⬇️ Download Your Analysis",
        user_df.to_csv(index=False),
        file_name=f"{city_name}_analysis.csv",
        mime="text/csv"
    )

if st.checkbox("Show raw data",key="show_user_data"):
    st.subheader("Raw Dataset")
    st.dataframe(user_df)
st.subheader("⬇️ Download Data")
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_air_pollution_data.csv",
    mime="text/csv"
)

if st.checkbox("Show raw data", key="show_raw_data_main"):
    st.subheader("Raw Dataset")
    st.dataframe(filtered_data)
