import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('RealWorldApplications/chargeback-fraud-detection-dataset.csv')

# defining X (features) and y (label)
X = np.array(df.drop(['CBK'], 1))
y = df['CBK']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# K=5, KNeighborsClassifier(n_neighbors=5, ...)
model = neighbors.KNeighborsClassifier()
model.fit(X_train, y_train)

# testing model accuracy
accuracy = model.score(X_test, y_test) 
accuracy

# Usage Example:
test_chargeback = np.array([[5396143442, 20150530223820, 448.5]])

is_fraud = model.predict(test_chargeback)
is_fraud