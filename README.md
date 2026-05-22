# 🎬 CineMatch — AI-Powered Movie Recommendation System

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Bebas+Neue&size=40&duration=3000&pause=1000&color=E50914&center=true&vCenter=true&width=600&lines=🎬+CineMatch;AI+Movie+Recommender;Search+%7C+Discover+%7C+Explore" alt="Typing SVG" />

---

[![Python](https://img.shields.io/badge/Python-3.10.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TMDB](https://img.shields.io/badge/TMDB-API-01B4E4?style=for-the-badge&logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

---

### 🚀 Live Demo

[![Live App](https://img.shields.io/badge/🎬_Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://movie-recommendation-system-rahul8878.streamlit.app)
[![API Docs](https://img.shields.io/badge/⚡_API_Docs-Swagger_UI-009688?style=for-the-badge)](https://movie-recommendation-system-6-t0c7.onrender.com/docs)
[![Health Check](https://img.shields.io/badge/✅_Health-Check_API-46E3B7?style=for-the-badge)](https://movie-recommendation-system-6-t0c7.onrender.com/health)

---

**A full-stack, production-ready movie recommender combining NLP-based content filtering with live TMDB data — served over a FastAPI backend and a beautiful Streamlit frontend.**

[Features](#-features) · [Architecture](#-architecture) · [How It Works](#-how-it-works) · [Setup Guide](#-step-by-step-setup-guide) · [API Reference](#-api-reference) · [Deployment](#-deployment) · [Contact](#-contact)

</div>

---

## 📌 Overview

**CineMatch** is an end-to-end movie recommendation engine built from scratch. It uses **TF-IDF vectorization** and **cosine similarity** on movie metadata to generate smart, content-based recommendations — then enriches results with **live poster images, ratings, and details** from The Movie Database (TMDB) API.

```
🧠 ML Engine  +  ⚡ FastAPI Backend  +  🎨 Streamlit Frontend  =  🎬 CineMatch
```

> Built as a **portfolio-grade** project demonstrating end-to-end ML engineering, REST API design, NLP, and cloud deployment.

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 🔍 | **Smart Search** | Search any movie — instant TMDB match with full details |
| 🤖 | **TF-IDF Recommendations** | Content-based filtering using cosine similarity on metadata |
| 🎭 | **Genre Discovery** | TMDB Discover API — find popular movies by genre |
| 🏠 | **Home Feed** | Browse Trending, Popular, Top Rated, Upcoming, Now Playing |
| 🖼️ | **Live Posters** | Every recommendation enriched with TMDB poster images |
| ⚡ | **Async API** | All TMDB calls made asynchronously using `httpx` |
| 🧩 | **Modular Design** | TF-IDF engine, TMDB client, and Streamlit UI fully decoupled |
| 🌐 | **CORS Ready** | Open CORS middleware for seamless frontend communication |
| 🚀 | **Cloud Deployed** | FastAPI on Render + Streamlit on Streamlit Cloud |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      🎨 Streamlit Frontend                       │
│        Home Feed │ Search Bar │ Movie Cards │ Recommendations    │
│     https://movie-recommendation-system-rahul8878.streamlit.app  │
└──────────────────────────┬───────────────────────────────────────┘
                           │  HTTP Requests
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ⚡ FastAPI Backend (main.py)                    │
│       https://movie-recommendation-system-6-t0c7.onrender.com   │
│                                                                  │
│  GET /home            →  TMDB trending / popular / top_rated    │
│  GET /tmdb/search     →  TMDB keyword search (multiple results) │
│  GET /movie/search    →  Bundle: Details + TF-IDF + Genre recs  │
│  GET /movie/id/{id}   →  TMDB movie details by ID               │
│  GET /recommend/tfidf →  Local TF-IDF content recommendations   │
│  GET /recommend/genre →  TMDB genre-based discovery             │
│  GET /health          →  API health check                       │
└─────────────────┬────────────────────────┬───────────────────────┘
                  │                        │
                  ▼                        ▼
   ┌──────────────────────┐    ┌──────────────────────────┐
   │  🧠 Local NLP Engine │    │    🌐 TMDB REST API       │
   │                      │    │  api.themoviedb.org/3    │
   │  df.pkl              │    │                          │
   │  tfidf_matrix.pkl    │    │  /search/movie           │
   │  tfidf.pkl           │    │  /movie/{id}             │
   │  indices.pkl         │    │  /discover/movie         │
   └──────────────────────┘    │  /trending/movie/day     │
                               └──────────────────────────┘
```

---

## 📁 Project Structure

```
📦 cinematch/
│
├── 🐍 main.py                  # FastAPI backend — all routes & business logic
├── 🎨 app.py                   # Streamlit frontend — UI components & API calls
│
├── 📓 movie.ipynb              # Jupyter notebook — data prep & TF-IDF training
│
├── 📊 df.pkl                   # Preprocessed movie DataFrame
├── 🔗 indices.pkl              # Title → row index mapping for fast lookup
├── 🤖 tfidf.pkl                # Fitted TF-IDF vectorizer
├── 🧮 tfidf_matrix.pkl         # Sparse TF-IDF document-term matrix
│
├── 📋 movies_metadata.csv      # Raw dataset (Kaggle Movies Dataset)
│
├── 📦 requirements.txt         # Python dependencies
├── 🐍 runtime.txt              # Python version (3.10.11)
├── 🔒 .env                     # Secret keys (NOT committed to Git)
└── 🚫 .gitignore               # Ignores .env, __pycache__, *.pkl
```

---

## 🧠 How It Works

### Step 1 — 📊 Data Preprocessing (`movie.ipynb`)

```python
# Extract relevant columns and build "soup" string per movie
df['soup'] = df['overview'] + ' ' + df['genres'] + ' ' + df['cast']
```

### Step 2 — 🤖 TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])
# Result: sparse matrix of shape (n_movies, n_features)
```

### Step 3 — 🎯 On-Demand Cosine Similarity

```python
# O(n) per query — much more efficient than O(n²) precomputation
query_vector = tfidf_matrix[movie_idx]
scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
top_indices = np.argsort(-scores)[1:top_n+1]  # exclude itself
```

### Step 4 — 🖼️ TMDB Enrichment

| Data | Source |
|------|--------|
| 🖼️ Poster image | TMDB `/w500` format |
| ⭐ Vote average | TMDB ratings |
| 📅 Release date | TMDB metadata |
| 📖 Overview | TMDB description |
| 🎭 Genres | TMDB genre list |

---

## ⚙️ Step-by-Step Setup Guide

### Prerequisites

- Python **3.10.11**
- A free [TMDB API key](https://www.themoviedb.org/settings/api)
- `pip` and `virtualenv`

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rahul07-github/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2️⃣ Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

```bash
# .env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 5️⃣ Start FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6️⃣ Start Streamlit Frontend

```bash
streamlit run app.py
```

### 7️⃣ Verify Everything Works

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

---

## 🔌 API Reference

### Base URL (Production)
```
https://movie-recommendation-system-6-t0c7.onrender.com
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | ✅ Health check |
| `GET` | `/home?category=popular` | 🏠 Home feed |
| `GET` | `/tmdb/search?query={q}` | 🔍 TMDB keyword search |
| `GET` | `/movie/id/{tmdb_id}` | 🎬 Movie details by ID |
| `GET` | `/movie/search?query={q}` | 📦 Bundle: Details + TF-IDF + Genre |
| `GET` | `/recommend/tfidf?title={t}` | 🤖 TF-IDF recommendations |
| `GET` | `/recommend/genre?tmdb_id={id}` | 🎭 Genre-based recommendations |

### Example Response

```json
{
  "query": "Interstellar",
  "movie_details": {
    "tmdb_id": 157336,
    "title": "Interstellar",
    "release_date": "2014-11-05",
    "poster_url": "https://image.tmdb.org/t/p/w500/...",
    "genres": [{"id": 12, "name": "Adventure"}]
  },
  "tfidf_recommendations": [
    {"title": "Gravity", "score": 0.412}
  ],
  "genre_recommendations": [...]
}
```

---

## 🔑 Key Technical Highlights

> 🔵 **Sparse Matrix Efficiency** — TF-IDF matrix stored as `scipy` sparse; O(n) per query, not O(n²)

> 🟢 **Async TMDB Client** — All external API calls use `httpx.AsyncClient` for non-blocking I/O

> 🟡 **Graceful Degradation** — Falls back to user query if title not in local dataset; never crashes

> 🔴 **Normalized Title Matching** — All lookups lowercased and stripped for consistency

> 🟣 **CORS Middleware** — Open CORS policy for seamless Streamlit ↔ FastAPI communication

---

## 📦 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10.11 |
| NLP / ML | scikit-learn, NumPy, SciPy | Latest stable |
| Data | Pandas | 2.2.2 |
| Backend API | FastAPI + Uvicorn | 0.111.0 |
| External Data | TMDB API + httpx | Async |
| Frontend | Streamlit | 1.36.0 |
| Persistence | Python pickle | Built-in |
| Config | python-dotenv | 1.0.1 |

---

## 🚀 Deployment

### 🔵 Backend → Render

```bash
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment Variables:
```
PYTHON_VERSION = 3.10.11
TMDB_API_KEY   = your_key_here
```

### 🔴 Frontend → Streamlit Cloud

```
Repository: rahul07-github/Movie-Recommendation-System
Branch:     master
Main file:  app.py
```

Secrets:
```toml
TMDB_API_KEY = "your_key_here"
```

---

## 🔮 Future Improvements

- [ ] 🤝 **Collaborative Filtering** — SVD matrix factorization for personalized recs
- [ ] 🔀 **Hybrid Recommender** — Blend TF-IDF + TMDB popularity scores
- [ ] 👤 **User Authentication** — Save favourites and watch history
- [ ] ⚡ **Redis Caching** — Cache TMDB responses to reduce API latency
- [ ] 🐳 **Docker Compose** — One-command containerized deployment
- [ ] 🧬 **Semantic Search** — Replace TF-IDF with sentence-transformers
- [ ] 📊 **Analytics Dashboard** — Track popular searches and trends

---

## 🙏 Acknowledgements

- [**TMDB**](https://www.themoviedb.org/) — Free movie API *(Not endorsed or certified by TMDB)*
- [**Kaggle Movies Dataset**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — Base training data
- [**FastAPI**](https://fastapi.tiangolo.com/) — Async Python API framework
- [**Streamlit**](https://streamlit.io/) — Python web app framework
- [**scikit-learn**](https://scikit-learn.org/) — TF-IDF vectorizer

---

## 📄 License

Licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 📬 Contact

**Rahul Kumar Jha**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/rahul07-github)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail)](mailto:your.email@gmail.com)

---

<div align="center">

[![Live App](https://img.shields.io/badge/🎬_Open_App-Click_Here-FF4B4B?style=for-the-badge)](https://movie-recommendation-system-rahul8878.streamlit.app)
[![API](https://img.shields.io/badge/⚡_API_Docs-Swagger-009688?style=for-the-badge)](https://movie-recommendation-system-6-t0c7.onrender.com/docs)
[![Repo](https://img.shields.io/badge/📂_Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/rahul07-github/Movie-Recommendation-System)

---

⭐ **If you found this project useful, please give it a star!** ⭐

*Made with ❤️ and lots of 🍿 by Rahul Kumar Jha*

</div>
