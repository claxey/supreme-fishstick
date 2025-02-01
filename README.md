# supreme-fishstick
Stock -market predictor using lstm
# Stock Market Prediction using LSTM and Streamlit

## Overview
This project is a stock market prediction system that utilizes **Long Short-Term Memory (LSTM)** networks to forecast stock prices. The application is built using **Streamlit** for both frontend and backend, providing an interactive UI for users to visualize predictions and sentiment analysis.

## Features
- **Stock Price Prediction:** Uses LSTM to analyze historical stock data and predict future trends.
- **Sentiment Analysis:** Analyzes market sentiment using NLP techniques.
- **Interactive UI:** Built with Streamlit for seamless user interaction.
- **Graphical Visualization:** Displays historical trends and predicted stock prices using Matplotlib and Plotly.
- **Real-time Data Fetching:** Retrieves stock data from Yahoo Finance or any provided dataset.

## Technologies Used
- **Python** (Core language)
- **Streamlit** (Frontend & Backend framework)
- **TensorFlow/Keras** (For LSTM implementation)
- **Pandas & NumPy** (Data preprocessing)
- **Matplotlib & Plotly** (Visualization)
- **NLTK & TextBlob** (Sentiment analysis)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/claxey/supreme-fishstick.git
   cd your-repo
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter the stock ticker symbol (e.g., AAPL, TSLA) in the input field.
- Click the **Predict** button to generate future stock price trends.
- View graphical representations of the stock's historical and predicted values.
- Perform sentiment analysis on stock-related news.

## Project Structure
```
📂 stock-market-predictor
│── app.py                 # Main Streamlit app
│── fetcher.py             # Fetches stock market data
│── preprocessing.py       # Data preprocessing scripts
│── train_lstm.py          # LSTM model training
│── predict.py             # Prediction script
│── sentiment_analysis.py  # Sentiment analysis implementation
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Screenshots
(![Screenshot 2025-02-01 232904](https://github.com/user-attachments/assets/73c874d0-1d54-44d5-aa08-b8d9132c15c3)
)
![Screenshot 2025-02-01 232919](https://github.com/user-attachments/assets/661701f0-deac-49a1-aeed-f4a55b232979)

![Screenshot 2025-02-01 232935](https://github.com/user-attachments/assets/968199a6-582c-4a19-9c09-b6e933e0293c)

![Screenshot 2025-02-01 232946](https://github.com/user-attachments/assets/c9e5babb-da50-490d-a7b4-718bc65c7e33)


## Future Improvements
- Adding more machine learning models for comparison.
- Implementing additional technical indicators.


## Contributing
Feel free to fork this project and submit a pull request with your improvements!

## License
MIT License © 2025 Ayush Chaudhary
