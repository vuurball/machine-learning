**Example:**

Predict the future Price of the currency pair

**Dataset:** 

Daily BTCUSD OHLC data, sorted by Date ASC

**Step 1: import libs and fetch the dataset**
```python
import pandas as pd
import math
import numpy as np #to work with arrays
from sklearn import preprocessing # for normalizing data pre model run
from sklearn.model_selection import train_test_split # for creating testing and training samples 
from sklearn.linear_model import LinearRegression 

import matplotlib.pyplot as plt #for charting the results
from matplotlib import style #for charting the results

style.use('ggplot') #for charting the results

# loading the dataset
df = pd.read_csv('BTCUSD.csv')
```

**Step 2: Defining the features of the dataset**

pushing all the columns/data into the classifier is not the best option 
we need to define the special relations between the columns, 
in other words create a more meaningful data using multiple column

```python
# so we will create a new column (High - Low) percent change column 
# * 100 is just for our readability
df['HighLowChangePercent'] = (df['High'] - df['Low']) / df['Low'] * 100.0

# and also a column daily percent change column  
df['DailyChangePercent'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# now we'll create a new df to work with, with only the important columns we need
# this is where we define the FEATURES
df= df[['Date', 'Close', 'HighLowChangePercent', 'DailyChangePercent', 'VolumeBTC', 'VolumeUSD']]
```

**Step 3: Preparing the data set**
```python
# in case there is missing data in the file, we will fill it with a default
# it is not possible to pass columns with NAN values, we have to replace them with some value
df.fillna(-99999, inplace=True)

# will will try to predict 1% day into the future:
# math.ceil() will round up a number to nearest full
# 0.01 * days we have in the dataset = days into the future of the prediction. i.e. the price in 20 days.
# the smaller the % is, the better the accuracy of the prediction
forecast_out = int(math.ceil(0.01 * len(df)))

# defining the label column that we want to predict
# so a new column is added (label) and it will have the prediction value for each row
# the shift -forcast_out (=20) means on each row the label will be a prediction of the price in 20 days
# so ie. on day 10 the label will show the price on day 30
df['label'] = df['Close'].shift(-forecast_out)
# up to this point we're not predicting yet, we only set the future price based on the data that is already in the dataset
# the label column will only show values in "label" column for the rows that have a +20 days price.
# so the last 20 rows will have a NaN, because the price in the real future is not in the dataset and needs to be predicted
```

**Step 4: Defining the FEATURES and the LABEL**

in our example the label is the price column "Close" 
because we want to try and predict the price in the future

```python
# features:
# get only the FEATURES columns without the Label col
X = np.array(df.drop(['label'], 1))

# scale X before passing it to classifier
# this step normalizes data, but if its done, it has to be done on all data, training, testing, and later on the real query data
X = preprocessing.scale(X)

# remove the rows that dont have value in Label (the last days in the set)
X = X[:-forecast_out ]

# the set of data we'll ask to predict for
X_to_predict = X[-forecast_out:]

# this line will remove the last rows that dont have predictions in the Label
df.dropna(inplace=True)

# label:
y = df['label']

# radomly splitting the X,y data sets into training 80% and testing 20% sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

```

**Step 5: creating the model**
```python
model = LinearRegression()

model.fit(X_train, y_train) # passing training X and y sets

score = model.score(X_test, y_test) #testing accuracy
```

**Step 6: using the model and making predictions**
```python
# forecast_set is an array of future prices predicted for each row in X_to_predict
forecast_set = model.predict(X_to_predict)
```

**Step 7: showing the prediction as a graph**
```python
df['Forecast'] = np.nan #setting default values to the new col

last_date = df.iloc[-1].name
last_date + 1

#setting the prediction value in the relevant rows
for i in forecast_set:
    df.loc[last_date+1] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,i]
    last_date= last_date+1

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.xlabel('price')
plt.show()
```
