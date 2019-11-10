########################################################################
# based on our dataset we want to see if what we can learn from it
# will the data have a clear pattern as to wether a person survives or not
# this kind of test could be run ahead of time to see what are the chance of survival of each person.
#
# dataset is info about titanic passangers (attached)
# including: passanger class, is_survived, name, gender, age, #siblings
# #parants/children, ticket #, ticket fare GBP, cabin, port, lifeboat, id, detination

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('K-MeansDataset.xls')

# droping useless columns
df.drop(['body', 'name'], 1, inplace=True)

df.fillna(0, inplace=True)

# most of the data is not numeric and we need to change that first
def handle_non_numerical_data(df):
    # array(['pclass', 'survived', 'sex', 'age',...
    columns = df.columns.values
    
    # looping on all columns
    for col in columns:
        
        # init dictionary for mapping string values to int values
        # i.e {"male":0, "female":1}
        text_digit_vals = {}
        
        def convert_to_int(val):
            return text_digit_vals[val]
        
        # if current column is not numeric:
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            #get all unique values in a column
            unique_values = set(column_contents) 
            x = 0
            # populate the dictinary
            for unique in unique_values: 
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            
            # replace the values using the dictionary
            df[col] = list(map(convert_to_int, df[col]))
    
    return df
    
df = handle_non_numerical_data(df)    

# removing the survived column because that's what we are testing
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = df['survived']

model = KMeans(n_clusters=2)
model.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = model.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

# result:        
(correct/len(X))

# the result value is flactuating between 0.7 and 0.2 which is the same as 70% 
# its a good result since we know we have 2 groups, "0"=died, "1"=survived
# we compare to those 2 groups, but the model might have classified things
# as "1"=died, "0"=survived
# so it looks like almost nothing matches, but in reality this means that the 
# 2 groups were sepparated correctly 70% of the time
 