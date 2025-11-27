# movie-recommendation-system-
🎬 Movie Recommendation System
A content-based movie recommender with posters, ratings, genres & trailers — powered by Python, Streamlit & TMDB API.
⭐ Overview

This project is a content-based movie recommendation system that suggests similar movies using TF-IDF vectorization and cosine similarity.
It integrates with the TMDB API to fetch live posters, ratings, genres, and YouTube trailers, delivering a clean, Netflix-style browsing experience inside a Streamlit app.

This project showcases machine learning, API integration, and web app development — making it an excellent addition to your developer portfolio.

🚀 Features

🔍 Search any movie from the dataset

🎯 Recommends top 5 similar movies

🖼️ Displays live posters from TMDB

⭐ Shows ratings, genre tags, and movie overview

▶️ Provides a "Watch Trailer" button

⚡ Fast and interactive Streamlit interface

🌐 Ready for deployment

🧠 How It Works

The system uses:

TF-IDF Vectorizer to convert movie descriptions into numerical vectors

Cosine Similarity to calculate how similar one movie is to another

TMDB API to fetch:

Posters

Ratings

Genres

Trailers

This combination of ML + live API data makes the recommendations both smart and visually appealing.

🛠️ Tech Stack

Frontend / UI: Streamlit
Backend: Python
Machine Learning: Scikit-learn (TF-IDF, Cosine Similarity)
Data: TMDB 5000 Movies Dataset
API: TMDB REST API (v3)
Other: Pandas, Requests

📁 Project Structure
movie-recommender/
│── app.py
│── movies.csv
│── credits.csv
│── requirements.txt
│── README.md
│── screenshots/
│    └── preview.png (optional)
