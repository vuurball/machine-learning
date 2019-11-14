import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('RealWorldApplications/travel-attractions-recommendation-dataset.csv', index_col=0)

df = df.fillna(0)
df.head(3)

# scaling all ratings to a smaller nummericals for more accuracy
X = preprocessing.scale(df)
X[:1]

# calculating the similarity % of each attraction to each other attraction
item_similarity = cosine_similarity(X.T)

# setting the results into a dataframe format for easy use
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)
item_similarity_df

def get_suggested_attractions(attraction, user_rating):
    # 2.5 is to make the results more extreme between what might be liked vs disliked
    attraction_scores = item_similarity_df[attraction] * (user_rating-2.5)
    attractions = attraction_scores.sort_values(ascending=False)
    return attractions


# Example 1:
print(get_suggested_attractions(attraction="The Jordaan", user_rating=5))    

# Example 2: 
new_user = [("The Jordaan", 5), ("Body Worlds", 1), ("Amsterdamse Bos", 1)]

suggestions = pd.DataFrame()

for attraction, rating in new_user:
    suggestions = suggestions.append(get_suggested_attractions(attraction, rating), ignore_index=True)
    
suggestions.sum().sort_values(ascending=False)    