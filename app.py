import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import logging

# Configure page
st.set_page_config(
    page_title="Stock Price Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Functions from previous code
@st.cache_data
def fetch_stock_data(ticker, end_date=None):
    if end_date is None:
        end_date = datetime.now().date()
    
    start_date = end_date - timedelta(days=10 * 365)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for ticker {ticker}.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_forecast_dataframe(forecast, start_date):
    try:
        forecast_dates = pd.date_range(start=start_date, periods=len(forecast), freq='B')
        forecast_df = pd.DataFrame({
            'Forecast_Date': forecast_dates,
            'Forecasted_Close_Price': forecast
        }).set_index('Forecast_Date')
        return forecast_df
    except Exception as e:
        st.error(f"Error creating forecast DataFrame: {str(e)}")
        return None

def forecast_stock(data, forecast_period=30, column='Close'):
    try:
        train = data[column]
        
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="mul",
            seasonal_periods=252,
            damped=True
        )
        fit = model.fit(optimized=True)
        forecast = fit.forecast(forecast_period)
        
        tomorrow = datetime.now().date() + timedelta(days=1)
        forecast_df = create_forecast_dataframe(forecast, tomorrow)
        
        return fit, forecast, forecast_df
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None

# Streamlit UI
def main():
    st.title("📈 Stock Price Forecasting App")
    st.write("Enter a stock ticker and forecast period to predict future stock prices.")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    forecast_days = st.sidebar.slider("Forecast Period (Days)", 5, 90, 30)
    
    # Add a "Generate Forecast" button
    if st.sidebar.button("Generate Forecast"):
        # Show loading message
        with st.spinner("Fetching stock data..."):
            data = fetch_stock_data(ticker)
        
        if data is not None:
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Historical Data")
                st.line_chart(data['Close'])
            
            # Perform forecasting
            with st.spinner("Generating forecast..."):
                model, forecast, forecast_df = forecast_stock(data, forecast_days)
            
            if forecast_df is not None:
                with col2:
                    st.subheader("Price Forecast")
                    st.line_chart(forecast_df['Forecasted_Close_Price'])
                
                # Show forecast data in a table
                st.subheader("Detailed Forecast")
                st.dataframe(forecast_df)
                
                # Add download button for forecast data
                csv = forecast_df.to_csv()
                st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"{ticker}_forecast.csv",
                    mime="text/csv"
                )
                
                # Show additional metrics
                st.subheader("Current Statistics")
                current_price = data['Close'].iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
                last_forecast = forecast_df['Forecasted_Close_Price'].iloc[-1]
                price_change = ((last_forecast - current_price) / current_price) * 100
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with metric_col2:
                    st.metric("Final Forecast", f"${last_forecast:.2f}")
                with metric_col3:
                    st.metric("Predicted Change", f"{price_change:.2f}%")

                
        else:
            st.error("Please enter a valid stock ticker.")

    # Add information about the app
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app uses historical stock data to forecast future prices using:
    - Historical data from Yahoo Finance
    - Exponential Smoothing for forecasting
    - Business day forecasting
    """)

if __name__ == "__main__":
    main()