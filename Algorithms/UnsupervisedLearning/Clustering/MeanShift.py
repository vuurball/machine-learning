########################################################################
# based on our dataset we want to see what we can learn from it
# will the data be split into clear groups, how many groups, and what features make the group 
# this kind of test could be run to categorize a set and understand which attributes in the data play an important role
#
# dataset is info about titanic passangers (attached)
# including: passanger class, is_survived, name, gender, age, #siblings
# #parants/children, ticket #, ticket fare GBP, cabin, port, lifeboat, id, detination
# the result of this example was a split into 3/4 groups. 1st group with mostly people who died, 
# 2nd where mostly all survived, and the rest are most of the other people.
# looking at the 1st&2nd group we can try to understand what makes the classifier group those people together, 
# meaning what can mean that a person had a great chance of surviving or not
# the conclution in not cut and dry, but it's mostly the passanger class, and might also be the number of children the person had

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('MeanShiftDataset.xls')
original_df = pd.DataFrame.copy(df)

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

model = MeanShift()
model.fit(X)

# get the groups the model created
labels = model.labels_

# adding new col to original df with textual values for readability
original_df['cluster_group'] = np.nan

# adding the group value to the new col for all rows in df
for i in range(len(X)):
    # iloc references the row at index i
    original_df['cluster_group'].iloc[i] = labels[i]
    
# the number of groups we got from the model    
n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    # creating a temp dataframe for category i only (this is a filter)
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    # filtering the df again to keep only those that survived
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    # calculating the rate of survivors in the i group
    survival_rate = len(survival_cluster)/len(temp_df)
    # adding the group -> survival rate to the dictionary
    survival_rates[i] = survival_rate 
    
#survival_rates    
#output->{0: 0.3798449612403101, 1: 1.0, 2: 0.1}

# to deeper analyze results, we can look at people in group 1 and see what they have in common that made them all survive
original_df[(original_df['cluster_group'] == 1)].describe()