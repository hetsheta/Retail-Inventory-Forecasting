import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import (
    load_data,
    preprocess_data,
    train_rf,
    forecast_inventory,
    check_inventory_alerts,
    save_model
)

st.set_page_config(page_title="Retail Inventory Forecasting", layout="wide")
st.title("📈 Retail Inventory Forecasting")

# Load & preprocess
data = load_data("data/retail_store_inventory.csv")
data = preprocess_data(data)

# Sidebar settings
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
alert_threshold = st.sidebar.number_input("Alert Threshold", min_value=1, value=100)

store_ids = sorted(data["Store ID"].unique())
categories = sorted(data["Category"].unique())

selected_store = st.sidebar.selectbox("Store ID", store_ids)
selected_category = st.sidebar.selectbox("Category", categories)

future_weather = st.sidebar.selectbox("Future Weather", ["Sunny", "Rainy", "Cloudy", "Snowy"])
future_promo = st.sidebar.selectbox("Future Holiday/Promotion (0=No, 1=Yes)", [0, 1])
future_price = st.sidebar.number_input("Future Price", min_value=0.0, value=50.0)

df_sc = data[(data["Store ID"] == selected_store) & (data["Category"] == selected_category)].copy()

if df_sc.empty:
    st.warning("No data for this store/category.")
    st.stop()

# Show data
if st.checkbox("Show Raw Data"):
    st.dataframe(df_sc)

# Train model
model, rmse = train_rf(df_sc)
save_model(model, f"models/model_{selected_store}_{selected_category}.pkl")

# Forecast
forecast_df = forecast_inventory(df_sc, model, forecast_days, future_weather, future_promo, future_price)
alerts = check_inventory_alerts(forecast_df, alert_threshold)

# 📈 Forecast only graph
st.subheader(f"Forecast for Store {selected_store} | Category {selected_category}")

forecast_part = forecast_df[forecast_df["Forecast"].notna()]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    forecast_part["Date"],
    forecast_part["Forecast"],
    label="Forecast",
    marker="o",
    linestyle="-",
    linewidth=2,
    color="tab:blue",
)
ax.set_title(f"Forecast Only — Next {forecast_days} Days", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Forecasted Inventory Level", fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# ⚠️ Alerts
if not alerts.empty:
    st.warning(f"⚠️ {len(alerts)} days forecasted below threshold {alert_threshold}!")
    
# 📅 Future Forecast Table with Sr_No
future_forecast = forecast_df[forecast_df["Inventory Level"].isna()][["Date", "Forecast"]].reset_index(drop=True)
future_forecast["Date"] = pd.to_datetime(future_forecast["Date"]).dt.strftime("%d-%m-%Y")
future_forecast.index += 1
st.subheader("Future Forecast")
st.dataframe(future_forecast)

# 📈 Actual + Forecast Line Chart
#st.subheader(f"Actual + Forecast | Store {selected_store} | Category {selected_category}")
#st.write(f"Model RMSE: {rmse:.2f}")
#st.line_chart(forecast_df.set_index("Date")[["Inventory Level", "Forecast"]])

# ✅ Download future forecast only with Sr_No
download_csv = future_forecast.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Future Forecast",
    download_csv,
    f"future_forecast_{selected_store}_{selected_category}.csv",
    "text/csv"
)
