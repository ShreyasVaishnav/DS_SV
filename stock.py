

import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go



START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predicition Web_App")
st.subheader("Model : FBProphet")
st.sidebar.header("Stock_List")

stocks = ("AAPL", "GOOGL", "MSFT", "GME", "TSLA")

selected_stocks = st.sidebar.selectbox("Select Dataset for Prediction", stocks)
n_years = st.slider("Years of Prediction:", 1, 5)
period = n_years * 365

@st.cache

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...DONE!")

st.subheader('Unfiltered Data fetched from Yahoo Finance')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'Stock_Open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Stock_Close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with FBProphet

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

'''
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
st.sidebar.header("Model_Accuracy")
R = r2_score(data.Close,forecast)
st.sidebar(R)
'''

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.write("Forecast components")
fig2 = m.plot_components(forecast)

st.write(fig2)
