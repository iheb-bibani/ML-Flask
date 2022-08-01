import joblib
import talib as ta
import yfinance as yf
from sklearn.svm import SVR

df = yf.download('MSFT', interval="1d")

df['sma'] = ta.SMA(df['Close'], 10)
df['rsi'] = ta.RSI(df['Close'], 14)

df.dropna(inplace=True)

X = df[['Open', 'sma', 'rsi']]
y = df['Close']

clf = SVR()
clf.fit(X.values, y)

joblib.dump(clf, 'model.sav')