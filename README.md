# 🎬 CineMatch — AI-Powered Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-API-01B4E4?style=for-the-badge&logo=themoviedatabase&logoColor=white)

**A full-stack, production-ready movie recommender that combines NLP-based content filtering with live TMDB data — served over a FastAPI backend and a beautiful Streamlit frontend.**

[Features](#-features) · [Architecture](#-architecture) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works) · [Setup Guide](#-step-by-step-setup-guide) · [API Reference](#-api-reference) · [Contact](#-contact)

</div>

---

## 📌 Overview

**CineMatch** is an end-to-end movie recommendation engine built from scratch. It leverages **TF-IDF vectorization** and **cosine similarity** on movie metadata to generate content-based recommendations — then enriches those results with **live poster images, ratings, and details** from The Movie Database (TMDB) API.

The system is architected as a **decoupled backend + frontend**:
- 🔧 **FastAPI** serves recommendations and TMDB-enriched data via a clean REST API
- 🎨 **Streamlit** provides a responsive, interactive UI with posters, search, and genre discovery
- 🧠 **NLP (TF-IDF + Cosine Similarity)** powers content-based filtering on movie overviews/metadata

> Built as a portfolio-grade project demonstrating end-to-end ML engineering, API design, and frontend deployment.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Smart Search** | Search any movie — returns best TMDB match with full details |
| 🤖 **Content-Based Recommendations** | TF-IDF cosine similarity on local dataset of thousands of movies |
| 🎭 **Genre Discovery** | TMDB Discover API fetches popular movies in the same genre |
| 🏠 **Home Feed** | Browse Trending, Popular, Top Rated, Upcoming, Now Playing |
| 🖼️ **Live Posters** | Every recommendation enriched with TMDB poster images |
| ⚡ **Fast Async API** | All TMDB calls made asynchronously with `httpx` |
| 🧩 **Modular Design** | TF-IDF engine, TMDB client, and Streamlit UI are fully decoupled |
| 🌐 **CORS Ready** | Open CORS middleware for seamless local + deployed frontend communication |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                      │
│   Home Feed │ Search Bar │ Movie Cards │ Recommendations    │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (main.py)                 │
│                                                             │
│  /home          →  TMDB trending / popular / top_rated     │
│  /tmdb/search   →  TMDB keyword search (multiple results)  │
│  /movie/search  →  Bundle: Details + TF-IDF + Genre recs   │
│  /movie/id/{id} →  TMDB movie details                      │
│  /recommend/tfidf   →  Local TF-IDF recommendations        │
│  /recommend/genre   →  TMDB genre-based discovery          │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼                      ▼
   ┌───────────────────┐   ┌──────────────────────┐
   │  Local NLP Engine │   │     TMDB REST API    │
   │                   │   │  api.themoviedb.org  │
   │  df.pkl           │   │                      │
   │  tfidf_matrix.pkl │   │  /search/movie       │
   │  tfidf.pkl        │   │  /movie/{id}         │
   │  indices.pkl      │   │  /discover/movie     │
   └───────────────────┘   │  /trending/movie/day │
                           └──────────────────────┘
```

---

## 📁 Project Structure

```
cinematch/
│
├── main.py                  # FastAPI backend — all routes & business logic
├── app.py                   # Streamlit frontend — UI components & API calls
│
├── movie.ipynb              # Jupyter notebook — data prep, TF-IDF model training
│
├── df.pkl                   # Preprocessed movie DataFrame (titles, metadata)
├── indices.pkl              # Title → row index mapping for fast lookup
├── tfidf.pkl                # Fitted TF-IDF vectorizer
├── tfidf_matrix.pkl         # Sparse TF-IDF document-term matrix
│
├── movies_metadata.csv      # Raw dataset (from TMDB / Kaggle Movies Dataset)
│
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version for deployment (Heroku / Render)
├── .env                     # Secret keys (NOT committed to Git)
└── .gitignore               # Ignores .env, __pycache__, *.pkl, etc.
```

---

## 🧠 How It Works

### Step 1 — Data Preprocessing (movie.ipynb)
Raw movie metadata (`movies_metadata.csv`) is cleaned and processed:
- Extract relevant columns: `title`, `overview`, `genres`, `cast`, `crew`, etc.
- Handle missing values, normalize text
- Build a **"soup"** string per movie (combined metadata for vectorization)

### Step 2 — TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])
# Result: sparse matrix of shape (n_movies, n_features)
```

### Step 3 — Cosine Similarity at Query Time
Instead of precomputing a full n×n similarity matrix (expensive), similarity is computed **on demand** per query:
```python
query_vector = tfidf_matrix[movie_idx]
scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
top_indices = np.argsort(-scores)[1:top_n+1]  # exclude itself
```

### Step 4 — TMDB Enrichment
Every recommended title is searched against the TMDB API to attach:
- 🖼️ Poster image (`/w500` format)
- ⭐ Vote average
- 📅 Release date
- 📖 Overview + genres

### Step 5 — Streamlit Frontend
The UI fetches from FastAPI endpoints, renders movie cards in responsive grids, and allows users to click any movie to see its full details and recommendations.

---

## ⚙️ Step-by-Step Setup Guide

### Prerequisites

- Python **3.10.11** (as specified in `runtime.txt`)
- A free TMDB API key
- `pip` and optionally `virtualenv`

---

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch
```

---

### 2. Create & Activate a Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# .env
TMDB_API_KEY=your_tmdb_api_key_here
```

---

### 5. Build the NLP Model (if pkl files are missing)

Open and run `movie.ipynb` in Jupyter:

```bash
pip install notebook
jupyter notebook movie.ipynb
```

This generates:
- `df.pkl` — cleaned movie DataFrame
- `indices.pkl` — title-to-index map
- `tfidf.pkl` — fitted TF-IDF vectorizer
- `tfidf_matrix.pkl` — sparse TF-IDF matrix

> ⚡ If the `.pkl` files are already included in the repo, skip this step.

---

### 6. Start the FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at: `http://localhost:7000`

Interactive docs: `http://localhost:8000/

---

### 7. Start the Streamlit Frontend

Open a **new terminal** (keep the FastAPI server running):

```bash
streamlit run app.py
```

The app will open at: `http://localhost:5885

---

### 8. Verify Everything Works

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# TF-IDF recommendation test
curl "http://localhost:8000/recommend/tfidf?title=Inception&top_n=5"

# Full search bundle
curl "http://localhost:8000/movie/search?query=The+Dark+Knight"
```

---

## 🔌 API Reference

### Base URL
```
http://localhost:8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/home?category=popular` | Home feed: `popular`, `trending`, `top_rated`, `upcoming`, `now_playing` |
| `GET` | `/tmdb/search?query={q}` | Raw TMDB keyword search (multiple results) |
| `GET` | `/movie/id/{tmdb_id}` | Full movie details by TMDB ID |
| `GET` | `/movie/search?query={q}` | **Bundle**: Details + TF-IDF recs + Genre recs |
| `GET` | `/recommend/tfidf?title={t}&top_n=10` | TF-IDF recommendations (local only) |
| `GET` | `/recommend/genre?tmdb_id={id}` | Genre-based recommendations via TMDB Discover |

### Example Response — `/movie/search?query=Interstellar`

```json
{
  "query": "Interstellar",
  "movie_details": {
    "tmdb_id": 157336,
    "title": "Interstellar",
    "overview": "The adventures of a group of explorers...",
    "release_date": "2014-11-05",
    "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "genres": [{"id": 12, "name": "Adventure"}, {"id": 18, "name": "Drama"}]
  },
  "tfidf_recommendations": [
    {"title": "Gravity", "score": 0.412, "tmdb": {"tmdb_id": 49047, "poster_url": "...", "vote_average": 7.1}}
  ],
  "genre_recommendations": [...]
}
```

---

## 🔑 Key Technical Highlights

- **Sparse Matrix Efficiency** — TF-IDF matrix is kept as a `scipy` sparse matrix; dot-product similarity is computed per query (O(n) not O(n²))
- **Async TMDB Client** — All external API calls use `httpx.AsyncClient` for non-blocking I/O in FastAPI
- **Graceful Degradation** — If a movie title isn't in the local dataset, the endpoint falls back to the user query; TMDB lookup failures return `None` instead of crashing
- **Normalized Title Matching** — All title lookups are lowercased and stripped to handle casing inconsistencies between TMDB and local dataset
- **CORS Middleware** — Open CORS policy allows seamless communication with the Streamlit frontend regardless of port

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| NLP / ML | scikit-learn (TF-IDF), NumPy, SciPy, Pandas |
| Backend API | FastAPI + Uvicorn |
| External Data | TMDB REST API + httpx (async) |
| Frontend | Streamlit |
| Model Persistence | Python `pickle` |
| Config Management | python-dotenv |
| Notebook | Jupyter |

---

## 🚀 Deployment

### Deploy to Render (Free Tier)

1. Push your repo to GitHub (make sure `.pkl` files are included or generated in `build.sh`)
2. Create a **Web Service** on [render.com](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `TMDB_API_KEY = your_key_here`

### Deploy Streamlit to Streamlit Cloud

1. Push `app.py` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy
4. Add `TMDB_API_KEY` in the Secrets panel
5. Update the FastAPI base URL in `app.py` to point to your Render deployment

---

## 🙏 Acknowledgements

- **[The Movie Database (TMDB)](https://www.themoviedb.org/)** — for their incredible free API providing movie metadata, posters, and discovery endpoints. *This product uses the TMDB API but is not endorsed or certified by TMDB.*
- **[Kaggle — The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)** — for the base `movies_metadata.csv` used to train the TF-IDF model
- **[FastAPI](https://fastapi.tiangolo.com/)** — for making async Python APIs a joy to build
- **[Streamlit](https://streamlit.io/)** — for turning Python scripts into shareable web apps effortlessly
- **[scikit-learn](https://scikit-learn.org/)** — for the robust and well-documented TF-IDF vectorizer
- Special thanks to the open-source Python ML community for the countless tutorials and discussions that shaped this project

---

## 🔮 Future Improvements

- [ ] **Collaborative Filtering** — Add user ratings + matrix factorization (SVD) for personalized recommendations
- [ ] **Hybrid Recommender** — Blend TF-IDF content scores with TMDB popularity scores
- [ ] **User Authentication** — Save favourite movies and watch history
- [ ] **Redis Caching** — Cache TMDB responses to reduce API calls and latency
- [ ] **Docker Compose** — Containerize FastAPI + Streamlit for one-command deployment
- [ ] **Semantic Search** — Replace TF-IDF with sentence-transformer embeddings for richer similarity

---

---

## 📬 Contact

**Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/rahuljha8878/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/rahul07-github)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat-square&logo=gmail)](rahulkumarjha9643@gmail.com)

> 💬 *Feel free to open an issue or reach out if you have questions, suggestions, or just want to talk movies!*

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

*Made with ❤️ and lots of 🍿*

</div>
