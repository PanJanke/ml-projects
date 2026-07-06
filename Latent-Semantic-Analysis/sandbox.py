import nltk
import plotly.express as px
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# Build a word-document matrix from book titles, use Truncated SVD to reduce
# word vectors to 2 dimensions, and visualize semantic relationships between
# words based on their co-occurrence in book titles.

nltk.download('punkt_tab') # Tokenizer models
nltk.download('stopwords') # List of stopwords
nltk.download('wordnet') # Lemmatizer models

wordnet_lemmatizer = WordNetLemmatizer()
titles = [line.rstrip('\n') for line in open('all_book_titles.txt')]
stops = set(stopwords.words('english'))
stops = stops.union({
    'introduction','edition','series','application','approach','card','access','package',
    'plus','etext','brief','vol','fundamental','guide','essential','printed','third','second',
    'fourth','fifth','sixth','volume'})

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2 and t not in stops]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

vectorizer = CountVectorizer(binary=True, tokenizer=my_tokenizer) # binary=True - we want only presence/absence of a word, not count
X = vectorizer.fit_transform(titles)
index_word_map = vectorizer.get_feature_names_out()
X=X.T
svd = TruncatedSVD()
Z = svd.fit_transform(X)

fig = px.scatter(x=Z[:,0], y=Z[:,1], text=index_word_map,size_max=60)
fig.update_traces(textposition='top center')
fig.show()