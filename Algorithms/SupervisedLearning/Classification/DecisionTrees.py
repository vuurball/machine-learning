import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.externals import joblib 
from sklearn import tree

music_data = pd.read_csv('music.csv') 
X = music_data.drop(columns=['genre']) 
y = music_data['genre'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #splitting 20% of data for testing set

# create model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# test model
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

# save model
joblib.dump(model, 'music-recomender.joblib') 

# visualize the model/ export to graph
tree.export_graphviz(model, out_file='model.dot', feature_names=['age','gender'], class_names=sorted(y.unique()), label="all", rounded=True, filled=True)

######################################################
#### importing and using the model inside an app: ####

from sklearn.externals import joblib # joblib has funcs for saving a loading models

model = joblib.load('music-recomender.joblib')
predictions = model.predict([[22,1], [22,0]])  

# output>array(['HipHop', 'Dance'], dtype=object)
