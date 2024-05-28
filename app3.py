import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define the date range
dt1 = "2014-01-01"
dt2 = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

# Define the stocks
stocks = {
    'Google': 'GOOGL',  # Alphabet Inc.
    'Microsoft': 'MSFT',  # Microsoft Corporation
    'Apple': 'AAPL',  # Apple Inc.
    'Nvidia': 'NVDA'  # Nvidia Corporation
}

selected_stock = st.selectbox('Select dataset for prediction', list(stocks.keys()))
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# Load data from Yahoo Finance
data_load_state = st.text('Loading data...')
data = yf.download(stocks[selected_stock], start=dt1, end=dt2)
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)
data_load_state.text('Loading data... done!')

# Display the raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Split the data into train and test
split_point = int(len(data) * 0.8)
train_data = data[:split_point]
test_data = data[split_point:]

# Prepare training data for Prophet
df_train = train_data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe and forecast
future = m.make_future_dataframe(periods=len(test_data))
forecast = m.predict(future)

# Extract forecast for the test period and calculate error metrics
forecast_test = forecast[-len(test_data):]

# Visualize the forecast vs actual
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Close'], name="Training Data"))
fig1.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name="Actual Test Data"))
fig1.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], name="Predicted Test Data"))
fig1.layout.update(title_text="Comparison of Predicted vs Actual Test Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

# Display error metrics
mae = mean_absolute_error(test_data['Close'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test_data['Close'], forecast_test['yhat']))
st.write('Mean Absolute Error on Test Data:', mae)
st.write('Root Mean Square Error on Test Data:', rmse)

# Forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

