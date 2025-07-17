import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # ✅ Filter only data from 2023
    df = df[df["Year"] == 2023]

    # Encode Weather Condition as numeric codes if needed
    if "Weather Condition" in df.columns:
        df["Weather Condition"] = df["Weather Condition"].astype("category").cat.codes

    return df


def train_rf(df):
    df = df.copy()
    df = df.sort_values("Date")

    # Define feature columns
    features = ["Year", "Month", "Day"]
    if "Weather Condition" in df.columns:
        features.append("Weather Condition")
    if "Holiday/Promotion" in df.columns:
        features.append("Holiday/Promotion")
    if "Price" in df.columns:
        features.append("Price")

    X = df[features]
    y = df["Inventory Level"]

    X_train, X_test = X.iloc[:-7], X.iloc[-7:]
    y_train, y_test = y.iloc[:-7], y.iloc[-7:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return model, rmse

def forecast_inventory(df, model, days, future_weather, future_promo, future_price):
    df = df.copy()
    df = df.sort_values("Date")
    last_date = df["Date"].max()

    # Encode weather
    weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Snowy": 3}
    weather_code = weather_map.get(future_weather, 0)
    promo_code = future_promo
    forecasts = []
    for i in range(1, days + 1):
        date = last_date + pd.Timedelta(days=i)
        features = [date.year, date.month, date.day, weather_code, promo_code, future_price]
        X_pred = np.array(features).reshape(1, -1)
        pred = model.predict(X_pred)[0]
        forecasts.append([date, pred])

    # Future
    future_df = pd.DataFrame(forecasts, columns=["Date", "Forecast"])

    # Historical actuals
    actual_df = df[["Date", "Inventory Level"]].copy()
    actual_df["Forecast"] = np.nan
    combined = pd.concat([actual_df, future_df], ignore_index=True)
    return combined

def check_inventory_alerts(forecast_df, threshold):
    future = forecast_df[forecast_df["Inventory Level"].isna()]
    alerts = future[future["Forecast"] < threshold]
    return alerts

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
