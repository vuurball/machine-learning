######################################################################################
# Classifiying class based on combination of attributes
# Dataset is attached, and taken from https://archive.ics.uci.edu/ml/datasets.php
######################################################################################

import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('K-NNdataset.data')

# replacing missing data '?' with a number '-99999'
df.replace('?', -99999, inplace=True)

# removing useless columns (like "ID")
# the accuracy of the model drops from ~96% to ~50% when this column is included 
df.drop(['id'], 1, inplace=True)

# defining X (features) and y (label)
X = np.array(df.drop(['class'],1))
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# K=5, KNeighborsClassifier(n_neighbors=5, ...)
model = neighbors.KNeighborsClassifier()
model.fit(X_train, y_train)

# testing model accuracy
accuracy = model.score(X_test, y_test) 

# just creating a row that doesnt exist in the dataset, without the last column
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,10,2,10,2,2,3,2,2]])

forecast_set = model.predict(example_measures)