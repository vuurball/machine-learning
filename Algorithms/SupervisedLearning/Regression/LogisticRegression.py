########################################################################
# Based on our dataset we want to see if we can predict wether a person survives or not

# Dataset is info about titanic passangers (attached)
# including: passanger class, is_survived, name, gender, age, #siblings
# #parants/children, ticket #, ticket fare GBP, cabin, port, lifeboat, id, detination


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn import preprocessing

df = pd.read_excel('LogisticRegressionDataset.xls')

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

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

# score = 95%~