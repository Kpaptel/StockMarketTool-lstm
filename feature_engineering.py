import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


ticker = '^GSPC'

#features = date, close, high, low, open, and volume
data = yf.download(ticker, start='2024-01-01', end='2025-01-01')
#get rid of multiIndex
data.columns = data.columns.droplevel(1)

#Calculation for Daily Return
#add new feature to dataset - use Return predict future movements
data['Return'] = data['Close'].pct_change()

#Feature added: SMA_10, SMA_50
#smoothes out noise by utilizing average and captures trends
data['SMA_10'] = data['Close'].rolling(10).mean()
data['SMA_50'] = data['Close'].rolling(50).mean()

#Feature added: RSI(Relative Strength Index) (14-day)
#momentum indicator
#below is calculation of rsi
delta = data['Close'].diff()
gain = (delta.where(delta>0,0)).rolling(14).mean()
loss = (-delta.where(delta<0,0)).rolling(14).mean()
RS = gain/loss
data['RSI'] = 100 - (100/(1+RS))

data['Volatility'] = data['Return'].rolling(10).std()

#clean data
data = data.dropna()

#set up the target variable
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Return', 'SMA_10', 'SMA_50', 'RSI', 'Volatility']

X = data[features].values
y = data['Target'].values.reshape(-1, 1)

#level out playing field for x values
#use mini minmax model to train on x and y
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

#use past 30 days
lookback = 30
X_seq, y_seq = [], []

#append sub arrays of n days(lookback period)
for i in range(lookback, len(X_scaled)):
    X_seq.append(X_scaled[i-lookback:i])
    y_seq.append(y_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)




# Check shapes
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)