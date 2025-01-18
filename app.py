import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
from datetime import datetime, timedelta
import io

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date=None, end_date=None):
    """Fetch stock data and properly handle MultiIndex columns."""
    if end_date is None:
        end_date = datetime.now().date()
    if start_date is None:
        start_date = end_date - timedelta(days=3 * 365)  # 3 years

    # Fetch stock data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Flatten the MultiIndex columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Keep only the first level of the MultiIndex

    return data

# Function to forecast using SARIMAX
def forecast_with_sarimax(data, forecast_period=30):
    try:
        model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 5))
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=forecast_period)
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()
        return forecast_mean, forecast_conf_int
    except Exception as e:
        st.error(f"Error in SARIMAX forecasting: {e}")
        return None, None

# Function to plot stock data and forecast
def plot_stock_data(data, forecast_mean, forecast_conf_int, forecast_period):
    fig = go.Figure()

    # Historical stock data plot
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data'))

    # Forecast plot
    forecast_dates = pd.date_range(data.index[-1], periods=forecast_period + 1, freq='B')[1:]
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='red')))

    # Confidence Interval
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_conf_int.iloc[:, 0], mode='lines', name='Lower CI', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_conf_int.iloc[:, 1], mode='lines', name='Upper CI', line=dict(color='red', dash='dash')))

    # Layout
    fig.update_layout(title='Stock Price Forecasting using SARIMAX', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Function for seasonal decomposition
def seasonal_decomposition(data):
    result = seasonal_decompose(data['Close'], model='additive', period=30)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(result.observed)
    axes[0].set_title("Observed")

    axes[1].plot(result.trend)
    axes[1].set_title("Trend")

    axes[2].plot(result.seasonal)
    axes[2].set_title("Seasonal")

    axes[3].plot(result.resid)
    axes[3].set_title("Residual")

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit layout
st.title("Stock Price Prediction Dashboard")
st.sidebar.title("Stock Settings")
st.warning("⚠️ **Important Disclaimer:** This application is for **educational purposes only**. Stock market investments are subject to **market risks**. Please consult a financial advisor before making any investment decisions.")
# Interactive selection for stock ticker and dates
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now().date())

# Fetch and display stock data
if ticker:
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is not None and not data.empty:
        st.write(f"Showing data for {ticker} from {start_date} to {end_date}")
        st.dataframe(data.tail())  # Show the last 5 rows of the data
        
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        seasonal_decomposition(data)

        # Forecasting section
        st.subheader("SARIMAX Forecast")
        forecast_period = st.slider("Select Forecast Period (days)", min_value=7, max_value=30, value=7)
        forecast_mean, forecast_conf_int = forecast_with_sarimax(data, forecast_period)

        if forecast_mean is not None:
            plot_stock_data(data, forecast_mean, forecast_conf_int, forecast_period)

            # Downloadable forecast data
            st.subheader("Download Forecast Data")
            forecast_df = pd.DataFrame({
                "Date": pd.date_range(data.index[-1], periods=forecast_period + 1, freq='B')[1:],
                "Forecast": forecast_mean,
                "Lower CI": forecast_conf_int.iloc[:, 0],
                "Upper CI": forecast_conf_int.iloc[:, 1]
            })
            csv = forecast_df.to_csv(index=False)
            st.download_button(label="Download Forecast Data as CSV", data=csv, file_name="forecast_data.csv", mime="text/csv")
    else:
        st.error("No data available for the selected ticker.")
