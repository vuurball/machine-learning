import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(i):
    return df.iloc[[i]].Title.values[0]


def get_index_from_title(title):
    try:
        return df[df.Title == title].index.values[0]
    except IndexError:
        return 0 # if not found
    
    
# putting more weight on films with same actors, than by genre or description
def concat_columns(row):
    return row['Actors'] +" "+ row['Actors'] +" "+ row['Description'] +" "+ row['Genre']

df = pd.read_csv('RealWorldApplications/movie-recommendation-dataset.csv')
df.head(2)    

df.fillna('', inplace=True)
          
# df.apply will run the concat_columns() func on each row (axis=1)
df['features'] = df.apply(concat_columns, axis=1)

model = CountVectorizer()
count_matrix = model.fit_transform(df['features'])

cosine_sim = cosine_similarity(count_matrix)
cosine_sim

# asking for recommendations of similar movies to "Mr. Brooks" film
similar_to_name = "Mr. Brooks"
similar_to_i = get_index_from_title(similar_to_name)

# getting row of cosine_sim for our movie above, this row will contain % of similarity to every other movie, in each col
# applying list(enumerate()) on the row, to get a list key:movie_id, value:similarity%
# that's how we preserve the movie id key
similar_movies = list(enumerate(cosine_sim[similar_to_i]))
similar_movies[:6]

# sorting the similar movies list by the similarity in descending order. 
# the keys are preserved to get the movie title from it
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
sorted_similar_movies[:6]

# printing out top 10 recommendations
i=1
for movie in sorted_similar_movies[1:11]:
    print(f"{i}: {get_title_from_index(movie[0])}")
    i+=1
 