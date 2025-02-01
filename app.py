import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
import requests
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import datetime
import joblib
from bs4 import BeautifulSoup

def fetch_news(symbol):
    api_key = "API_KEY"
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        news_articles =[(article["title"],article["url"]) for article in articles[:5]]  # Get top 5 news headlines
        return news_articles
    return []

def analyze_sentiment(texts):
    sentiment_scores = [TextBlob(text[0]).sentiment.polarity for text in texts]
    return sentiment_scores


def generate_recommendation(sentiment_scores):
    if not sentiment_scores:
        return "No sentiment data available"
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    if avg_sentiment > 0.2:
        return "📈 Strong Buy"
    elif avg_sentiment > 0:
        return "🔹 Optional Buy"
    elif avg_sentiment < -0.2:
        return "📉 Strong Sell"
    else:
        return "🔸 Optional Sell"


model = tf.keras.models.load_model("models/lstm_model.h5")
scaler = joblib.load("data/scaler.pkl")


def get_stock_data(symbol, start_date):
    df = yf.download(symbol, start=start_date)
    return df


def get_stock_summary(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "Company": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Stock Logo": info.get("logo_url", "")
        }
    except:
        return {}


def predict_stock(df):
    if df.empty:
        return None

    data = df[['Close']].values  
    data_scaled = scaler.transform(data)

    SEQ_LENGTH = 60
    X_input = data_scaled[-SEQ_LENGTH:]
    X_input = np.expand_dims(X_input, axis=0)

    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor & Sentiment Analysis")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")

stock_summary = get_stock_summary(symbol)
if stock_summary:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📌 {stock_summary.get('Company', 'N/A')}")
        st.write(f"**Sector:** {stock_summary.get('Sector', 'N/A')}")
        st.write(f"**Industry:** {stock_summary.get('Industry', 'N/A')}")
        st.write(f"**Market Cap:** {stock_summary.get('Market Cap', 'N/A')}")
        st.write(f"**P/E Ratio:** {stock_summary.get('P/E Ratio', 'N/A')}")
        st.write(f"**52 Week High:** ${stock_summary.get('52 Week High', 'N/A')}")
        st.write(f"**52 Week Low:** ${stock_summary.get('52 Week Low', 'N/A')}")
    with col2:
        if stock_summary.get("Stock Logo"):
            st.image(stock_summary["Stock Logo"], width=150)


start_date = st.date_input("Select Start Date:", datetime.date(2019, 1, 1))
days_to_display = st.slider("Show data for the last N days:", min_value=5, max_value=365, value=30)


if st.button("Fetch Data"):
    df = get_stock_data(symbol, start_date)

    if df.empty:
        st.error("⚠️ No data found! Please enter a valid stock symbol.")
    else:
        st.success(f"✅ Data fetched for {symbol}")

        
        st.subheader(f"📉 {symbol} Stock Price Chart")
        st.line_chart(df["Close"].tail(days_to_display))

       
        st.write("📊 **Recent Stock Data:**")
        st.dataframe(df.tail(5))

       
        st.subheader("📰 News Sentiment Analysis")
        news_articles = fetch_news(symbol)
        if news_articles:
            sentiment_scores = analyze_sentiment(news_articles)
            recommendation = generate_recommendation(sentiment_scores)

            for i, article in enumerate(news_articles):
                st.write(f"🔹 {i+1}. {article}")

            st.subheader("📌 Sentiment-based Recommendation:")
            st.markdown(f"**{recommendation}**")

        else:
            st.warning("⚠️ No news articles found for this stock.")

        st.subheader("📈 Predicted Stock Price")
        predicted_price = predict_stock(df)
        if predicted_price:
            st.success(f"📌 **Predicted Closing Price for Next Day:** **${predicted_price:.2f}**")
        else:
            st.error("⚠️ Not enough data to make a prediction.")

