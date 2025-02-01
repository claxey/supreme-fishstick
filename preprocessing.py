import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("data\stock_data.csv", index_col="Date", parse_dates=True)

data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0, 1))   #normalizing data:to 0 and 1 input......we do it by x_scaled=(x-minimum_val(x))/(max_val_of_x-min_val(x)) 

 
data_scaled = scaler.fit_transform(data)       #scale_fit= finds the min and max value in the data_set "data"


                                               #scaler_transform each value using min_max
                                               
def create_sequences(data, seq_length=60):    #made a sliding window for input of sequential data
    X, y = [], []                             #takin two array : 1) x(input) contains closing prices of 60 previous days 
                                              #                2) y(output)  comtains closing prices of next 60 days  
    for i in range(len(data) - seq_length):     
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
train_size = int(len(data_scaled) * 0.8)    #taking 80% of the data set to train
 
train_data = data_scaled[:train_size]        #fiting the training data set
test_data = data_scaled[train_size:]         #fitting the testing data set 

X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

np.save("data\X_train.npy", X_train)
np.save("data\y_train.npy", y_train)
np.save("data\X_test.npy", X_test)
np.save("data\y_test.npy", y_test)
joblib.dump(scaler, 'data\scaler.pkl')
