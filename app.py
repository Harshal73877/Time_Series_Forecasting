import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import datetime

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to forecast using SARIMAX
def forecast_with_sarimax(data, forecast_period=30):
    # Get the SARIMAX parameters, here you can adjust them as needed
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
    forecast_dates = pd.date_range(data.index[-1], periods=forecast_period+1, freq='B')[1:]
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='red')))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_conf_int.iloc[:, 0], mode='lines', name='Lower CI', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_conf_int.iloc[:, 1], mode='lines', name='Upper CI', line=dict(color='red', dash='dash')))
    
    # Layout
    fig.update_layout(title='Stock Price Forecasting using SARIMAX', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Streamlit layout
st.title("Stock Price Dashboard with SARIMAX Forecasting")

# Interactive selection for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

# Fetch and display stock data
if ticker:
    data = get_stock_data(ticker, start_date, end_date)
    
    if data is not None and not data.empty:
        st.write(f"Showing data for {ticker} from {start_date} to {end_date}")
        st.dataframe(data.tail())  # Show latest 5 rows of data

        # Forecasting section
        forecast_period = st.slider("Select Forecast Period (days)", min_value=7, max_value=30, value=7)
        
        forecast_mean, forecast_conf_int = forecast_with_sarimax(data, forecast_period)
        
        if forecast_mean is not None:
            plot_stock_data(data, forecast_mean, forecast_conf_int, forecast_period)
    else:
        st.write("No data available for the selected ticker.")
