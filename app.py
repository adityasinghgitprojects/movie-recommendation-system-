import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# --------------------------------------------------------
# 🔑 Your TMDB API Key
# --------------------------------------------------------
API_KEY = "0a32930b2133cfe6bf01f64b4b95b8fa"  # ← Replace with your real TMDB key

# --------------------------------------------------------
# 📂 Load movie dataset
# --------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    return movies

# --------------------------------------------------------
# ⚙️ Build similarity matrix
# --------------------------------------------------------
@st.cache_data
def build_similarity_matrix(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    movies["overview"] = movies["overview"].fillna("")
    matrix = tfidf.fit_transform(movies["overview"])
    return cosine_similarity(matrix)

# --------------------------------------------------------
# 🎯 Get top 5 similar movies
# --------------------------------------------------------
@st.cache_data
def get_recommendations(title, movies, sim_matrix):
    idx = movies[movies["title"].str.lower() == title.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in scores[1:6]]
    return movies.iloc[rec_indices]["title"].tolist()

# --------------------------------------------------------
# 🖼️ Fetch poster, rating, genres, year, and trailer
# --------------------------------------------------------
def fetch_movie_info(movie_title):
    query = urllib.parse.quote(movie_title)
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"

    try:
        res = requests.get(search_url)
        data = res.json()
        if data.get("results"):
            result = data["results"][0]
            movie_id = result["id"]
            poster_path = result.get("poster_path")
            rating = result.get("vote_average", "N/A")
            release_date = result.get("release_date", "")
            year = release_date.split("-")[0] if release_date else "N/A"

            # Fetch genre names
            genres_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            g_data = requests.get(genres_url).json()
            genres = [g["name"] for g in g_data.get("genres", [])]
            genre_text = ", ".join(genres) if genres else "N/A"

            # Fetch trailer
            trailer_url = None
            trailer_endpoint = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
            t_data = requests.get(trailer_endpoint).json()
            if t_data.get("results"):
                for vid in t_data["results"]:
                    if vid["type"] == "Trailer" and vid["site"] == "YouTube":
                        trailer_url = f"https://www.youtube.com/watch?v={vid['key']}"
                        break

            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450.png?text=No+Poster"
            return {
                "poster": poster_url,
                "rating": rating,
                "year": year,
                "genres": genre_text,
                "trailer": trailer_url,
            }

    except Exception as e:
        print("Error fetching info:", e)

    return {"poster": "https://via.placeholder.com/300x450.png?text=No+Poster", "rating": "N/A", "year": "N/A", "genres": "N/A", "trailer": None}

# --------------------------------------------------------
# 🧠 Streamlit App UI
# --------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")
st.title("🎥 Movie Recommendation System")
st.caption("Discover similar movies with posters, ratings, genres, and trailers!")

movies = load_data()
sim_matrix = build_similarity_matrix(movies)
movie_list = movies["title"].values

selected = st.selectbox("Search for a movie:", movie_list)

if st.button("Recommend"):
    recs = get_recommendations(selected, movies, sim_matrix)
    if not recs:
        st.warning("Movie not found in database.")
    else:
        st.subheader("Recommended Movies:")
        cols = st.columns(5)
        for i, title in enumerate(recs):
            info = fetch_movie_info(title)
            with cols[i]:
                st.image(info["poster"], use_container_width=True)
                st.markdown(f"**🎬 {title} ({info['year']})**")
                st.caption(f"⭐ {info['rating']} | 🎭 {info['genres']}")
                if info["trailer"]:
                    st.markdown(f"[▶ Watch Trailer]({info['trailer']})", unsafe_allow_html=True)
                else:
                    st.markdown("<small>No trailer available</small>", unsafe_allow_html=True)
