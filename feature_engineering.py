import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


ticker = '^GSPC'

#features = date, close, high, low, open, and volume
data = yf.download(ticker, start='2004-01-01', end='2025-01-01')
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


#beginning of the lstm pipeline, each layer represents lstm computation
model = Sequential()
#50 memory cells to keep track of past inputs
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#prevent relying too much on a memory cell
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


# Predict scaled values
y_pred_scaled = model.predict(X_test)

# Convert predictions back to original price scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)


# Check shapes
# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_val:", X_val.shape)
# print("y_val:", y_val.shape)
# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)

mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_actual, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

plt.figure(figsize=(12,6))
plt.plot(y_actual, color='blue', label='Actual Price')
plt.plot(y_pred, color='red', label='Predicted Price')
plt.title('S&P 500 Price Prediction')
plt.xlabel('Test Sequence')
plt.ylabel('Price')
plt.legend()
plt.show()