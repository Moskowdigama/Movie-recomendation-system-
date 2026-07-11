import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))

# Recompute similarity
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(matrix, matrix)

# Recommend function
def recommend(title):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:20]
    results = []
    for i, score in scores:
        combined = score * 0.6 + movies['imdb_norm'].iloc[i] * 0.3 + movies['pop_norm'].iloc[i] * 0.1
        results.append((i, combined))
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    return [(movies['title'].iloc[i], round(movies['imdb_score'].iloc[i], 1)) for i, _ in results]

# UI
st.title('🎬 Movie Recommender')
selected = st.selectbox('Pick a movie', movies['title'].values)

if st.button('Recommend'):
    results = recommend(selected)
    st.subheader('Top 5 Recommendations:')
    for title, score in results:
        st.write(f"**{title}** — IMDB: {score}")
