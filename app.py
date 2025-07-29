
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="NSE/BSE Stock Predictor", layout="wide")
st.title("📈 NSE/BSE Stock Growth Predictor")

# User input
symbol = st.text_input("Enter NSE/BSE stock symbol (e.g., RELIANCE.NS, TCS.NS):", "RELIANCE.NS")

# Date range
end = date.today()
start = end - timedelta(days=365)

# Fetch data
try:
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        st.warning("No data found. Please check the symbol.")
    else:
        st.success("Data fetched successfully!")

        # Show data
        st.subheader("Recent Stock Data")
        st.dataframe(data.tail(10))

        # Plot closing price
        st.subheader("Closing Price Chart")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Technical Indicators
        st.subheader("Technical Indicators")
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).mean()))

        fig2, ax2 = plt.subplots()
        ax2.plot(data['Close'], label='Close')
        ax2.plot(data['SMA_20'], label='20-Day SMA', linestyle='--')
        ax2.plot(data['SMA_50'], label='50-Day SMA', linestyle='--')
        ax2.set_title('Moving Averages')
        ax2.legend()
        st.pyplot(fig2)

        st.line_chart(data['RSI'].dropna(), height=150)

        # Simple Prediction: Linear Regression
        st.subheader("Next 7 Days Prediction (Linear Trend Model)")
        data = data.reset_index()
        data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

        X = data['Date_Ordinal'].values.reshape(-1, 1)
        y = data['Close'].values

        model = LinearRegression()
        model.fit(X, y)

        future_dates = [end + timedelta(days=i) for i in range(1, 8)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_preds = model.predict(future_ordinals)

        # Flatten prediction
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price": future_preds.ravel()
        })
        st.dataframe(pred_df)

        fig3, ax3 = plt.subplots()
        ax3.plot(data['Date'], data['Close'], label='Historical')
        ax3.plot(pred_df['Date'], pred_df['Predicted Close Price'], label='Prediction', linestyle='--')
        ax3.set_title("Historical & Predicted Prices")
        ax3.legend()
        st.pyplot(fig3)

        # Buy/Sell signal
        last_price = data['Close'].iloc[-1]
        # Buy/Sell signal last_price = data['Close'].iloc[-1] predicted_price = future_preds.ravel()[-1]
        if predicted_price > last_price * 1.03:
            st.success(f"📈 Suggestion: BUY — Predicted growth of {predicted_price - last_price:.2f} INR")
        elif predicted_price < last_price * 0.97:
            st.error(f"📉 Suggestion: SELL — Predicted drop of {last_price - predicted_price:.2f} INR")
        else:
            st.info("⏸️ Suggestion: HOLD — No major price movement expected")

except Exception as e:
    st.error(f"Error fetching or processing data: {e}")
