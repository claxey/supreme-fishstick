# sentiment_analysis.py

import requests
from textblob import TextBlob
from bs4 import BeautifulSoup

# Fetch news based on stock symbol
def fetch_news(symbol):
    url = f"https://www.google.com/search?q={symbol}+stock+news"  
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        headlines = soup.find_all("h3")
        
        news_articles = []
        for headline in headlines:
            news_articles.append(headline.get_text())
        
        return news_articles
    else:
        return None

def analyze_sentiment(news_articles):
    sentiment_scores = []
    
    for text in news_articles:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        
        sentiment_scores.append(sentiment)
    
    return sentiment_scores

def generate_recommendation(sentiment_scores):
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    if avg_sentiment > 0.5:
        return "Strong Buy"
    elif avg_sentiment > 0:
        return "Optional Buy"
    elif avg_sentiment < -0.5:
        return "Sell"
    else:
        return "Hold"
