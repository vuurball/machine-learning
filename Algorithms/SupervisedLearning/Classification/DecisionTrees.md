example of simple model training and predictions using a decision tree algo
for predicting a persons music genre preference based on age and gender:

![concept](https://d1rwhvwstyk9gu.cloudfront.net/2019/02/Decision-Trees.jpg)

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
    
music_data = pd.read_csv('music.csv') //scv with cols AGE, GENDER, GENRE 
X = music_data.drop(columns=['genre']) //leaving the attributes columns only
y = music_data['genre'] //selecting the result column obly

model = DecisionTreeClassifier()
model.fit(X, y) //training a model with set of attributes (X) and a set of results (y)
predictions = model.predict([[22,1], [22,0]]) //asking for 2 predictions based on 2 cols (age and gender)
predictions 

output> array(['HipHop', 'Dance'], dtype=object)
```

example of how to split data set for training and testing, and test prediction accuracy
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # for model algorithm
from sklearn.model_selection import train_test_split # for spliting dataset
from sklearn.metrics import accuracy_score # for testing prediction accuracy

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #splitting 20% of data for testing set

model = DecisionTreeClassifier()
model.fit(X_train, y_train) # passing training X and y sets
predictions = model.predict(X_test) #passing test attributes only
score = accuracy_score(y_test, predictions) #checking prediction accuracy
score # 1.0 means 100% accuracy
#run code few times because dataset is split randomely and might cause different score each run
```

once we are happy with our model's predictions score we need to save the model so we can use it
run this code to create model, train it, and then save it, (location as the jupyter notebook file)
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib # joblib has funcs for saving a loading models
    
music_data = pd.read_csv('music.csv') 
X = music_data.drop(columns=['genre'])
y = music_data['genre'] 

model = DecisionTreeClassifier()
model.fit(X, y) 

joblib.dump(model, 'music-recomender.joblib') #saving into file
```

now we can load the model file we created, to use it for predictions in our app
```python
from sklearn.externals import joblib # joblib has funcs for saving a loading models

model = joblib.load('music-recomender.joblib')

predictions = model.predict([[22,1], [22,0]]) 
predictions 

output>array(['HipHop', 'Dance'], dtype=object)
```

to visualize our decision model in a graph
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree # this obj has method for exporting model in graphical format

music_data = pd.read_csv('music.csv') 
X = music_data.drop(columns=['genre']) 
y = music_data['genre'] 

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(model, 
                     out_file='music-recommender.dot',
                     feature_names=['age','gender'],
                     class_names=sorted(y.unique()),
                     label="all", #graph will have labels we can read
                     rounded=True, #graph nodes will have rounded corners
                     filled=True #graph nodes will have bg color filling
                    )
```
the file `music-recommender.dot` was created in the dir of the jupyter notebook
open it in visual studio, and install extention to view it
go to extentions, search for `dot` and select `Graphviz (dot) language..` install it and view the graph
