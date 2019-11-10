import pandas as pd
import math
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_csv('BTCUSD.csv')

df['HighLowChangePercent'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['DailyChangePercent'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
df= df[['Date', 'Close', 'HighLowChangePercent', 'DailyChangePercent', 'VolumeBTC', 'VolumeUSD']]
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df['Close'].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out ]
X_to_predict = X[-forecast_out:]

df.dropna(inplace=True)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LinearRegression()
model.fit(X_train, y_train) # passing training X and y sets

# test model
score = model.score(X_test, y_test) #testing accuracy

# use the model
forecast_set = model.predict(X_to_predict)

# visualize results
df['Forecast'] = np.nan #setting default values to the new col

last_date = df.iloc[-1].name
last_date + 1
for i in forecast_set:
    df.loc[last_date+1] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,i]
    last_date= last_date+1

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.xlabel('price')
plt.show()

