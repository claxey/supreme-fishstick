import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock_symbol=input("enter the ticker yu want to check:- ")
df= yf.download(stock_symbol, start="2019-01-01", end="2024-01-01")
df.to_csv("data\stock_data.csv")

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label=f'{stock_symbol} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
