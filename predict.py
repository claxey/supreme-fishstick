import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
model = tf.keras.models.load_model("models\lstm_model.h5")
scaler = joblib.load("data\scaler.pkl")

X_test = np.load("data\X_test.npy")
y_test = np.load("data\y_test.npy")

predicted_prices = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

df = pd.read_csv("data\stock_data.csv", index_col="Date", parse_dates=True)
df = df.iloc[-len(predicted_prices):]
df["Predicted_Close"] = predicted_prices

df.to_csv("data\predicted_stock_prices.csv")

print(df.tail())
