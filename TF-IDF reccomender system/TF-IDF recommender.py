import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def genres_to_string(row):
  genres = row['Genres']
  genres = ''.join(''.join(j.split(' ')) for j in genres)
  return genres

df = pd.read_csv('data/games.csv')
df['Title'] = df['Title'].str.strip()
df = df.drop_duplicates(subset='Title')
df['string'] = df.apply(genres_to_string, axis=1)
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['string'])
print(f"Wymiary macierzy TF-IDF: {X.shape[0]} dokumentów, {X.shape[1]} cech")
print(f"Liczba niezerowych elementów: {X.nnz}")

game2index = pd.Series(df.index, index=df['Title'])

def recommend(title):
  idx = game2index[title]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]

  # calculate the pairwise similarities for this movie
  query = X[idx]
  scores = cosine_similarity(query, X)

  # currently the array is 1 x N, make it just a 1-D array
  scores = scores.flatten()

  recommended_idx = (-scores).argsort()[1:6]

  # return the titles of the recommendations
  return df[['Title','Genres']].iloc[recommended_idx]

print("Recommendations for 'Portal 2':")
print(recommend('Portal 2'))

print(game2index['Portal 2'])
query = X[game2index['Portal 2']]
scores = cosine_similarity(query, X)
scores = scores.flatten()
(-scores).argsort()
plt.plot(scores[(-scores).argsort()])
plt.show()

## Wymiary macierzy TF-IDF: 1512 dokumentów, 27 cech
## Liczba niezerowych elementów: 3754
## Recommendations for 'Elden Ring ':
##                                   Title                Genres
## 356                      Dark Souls III  ['Adventure', 'RPG']
## 0                            Elden Ring  ['Adventure', 'RPG']
## 352              Xenoblade Chronicles 3  ['Adventure', 'RPG']
## 38                       Genshin Impact  ['Adventure', 'RPG']
## 1348  The Witcher 2: Assassins of Kings  ['Adventure', 'RPG']

##^ źle dobrany dataset, jest tylko 27 unikalnych gatunków, przez co top 5
# ( dla Elden Ring ponad top 400 - tylko dwa tagi) jest identyczna - waga 1.0
## po usunięciu duplikatów Liczba niezerowych elementów: 2731

## Recommendations for 'Portal 2 ':
##                           Title                                          Genres
## 13                     Portal 2  ['Adventure', 'Platform', 'Puzzle', 'Shooter']  <- jest tu portal bo jest dużo tytułów ze scorem 1.0, więc
## 727   Half-Life: Opposing Force               ['Platform', 'Puzzle', 'Shooter']          recommended_idx = (-scores).argsort()[1:6]
## 18                       Portal               ['Platform', 'Puzzle', 'Shooter']                                                ^ nie zadziała
## 1358           Prince of Persia             ['Adventure', 'Platform', 'Puzzle']
## 681                Wario Land 3             ['Adventure', 'Platform', 'Puzzle']
