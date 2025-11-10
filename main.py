import yfinance as yf
import pandas as pd

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


print(data.head())