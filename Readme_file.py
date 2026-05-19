## Run this file- cd "Project NLP"
## Step 2: uvicorn main:app --reload
## pip install -r requirements.txt
##.\.venv\Scripts\Activate.ps1


import requests
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================
API_BASE = "https://movie-rec-466x.onrender.com" or "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES — Dark Cinema Theme
# =============================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Mulish:wght@300;400;600;700;800&display=swap');

/* ── ROOT VARIABLES ─────────────────────────── */
:root {
  --bg:         #08080f;
  --bg2:        #0f0f18;
  --bg3:        #16161f;
  --bg4:        #1e1e2c;
  --accent:     #e50914;
  --accent-dim: rgba(229,9,20,0.18);
  --gold:       #c9a84c;
  --gold-dim:   rgba(201,168,76,0.15);
  --text:       #e8e8f0;
  --text-dim:   #a0a0bc;
  --muted:      #58587a;
  --border:     rgba(255,255,255,0.06);
  --border-red: rgba(229,9,20,0.35);
  --card-bg:    rgba(16,16,26,0.97);
  --shadow-lg:  0 16px 48px rgba(0,0,0,0.75);
  --glow-red:   0 0 28px rgba(229,9,20,0.45);
  --glow-gold:  0 0 22px rgba(201,168,76,0.38);
  --radius:     10px;
  --radius-lg:  18px;
  --radius-xl:  24px;
}

/* ── GLOBAL RESET ───────────────────────────── */
* { box-sizing: border-box; }

/* ── APP BACKGROUND ─────────────────────────── */
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 90% 45% at 50% -5%,  rgba(229,9,20,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 55% 35% at 85% 90%,  rgba(201,168,76,0.04) 0%, transparent 55%),
    radial-gradient(ellipse 40% 30% at 5% 50%,   rgba(229,9,20,0.03) 0%, transparent 50%) !important;
  font-family: 'Mulish', sans-serif !important;
  color: var(--text) !important;
}

/* ── BLOCK CONTAINER ────────────────────────── */
.block-container {
  padding-top: 1.5rem   !important;
  padding-bottom: 3rem  !important;
  max-width: 1500px     !important;
}

/* ══════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b0b14 0%, #0d0d18 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
  font-family: 'Mulish', sans-serif !important;
}
[data-testid="stSidebar"] h2 {
  font-family: 'Bebas Neue', sans-serif !important;
  letter-spacing: 4px !important;
  font-size: 1.55rem !important;
  color: #fff !important;
  margin-bottom: 0.2rem !important;
}
[data-testid="stSidebar"] h3 {
  font-family: 'Mulish', sans-serif !important;
  font-size: 0.72rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
/* Sidebar home button */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-dim) !important;
  border-radius: var(--radius) !important;
  width: 100% !important;
  font-family: 'Mulish', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.5px !important;
  padding: 0.55rem 1rem !important;
  transition: all 0.22s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  box-shadow: var(--glow-red) !important;
  transform: translateX(5px) !important;
}
/* Sidebar selectbox */
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--bg4) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}
/* Slider accent */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stSliderTrack"] {
  background: var(--accent) !important;
}

/* ══════════════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════════════ */
h1, h2 {
  font-family: 'Bebas Neue', sans-serif !important;
  letter-spacing: 4px !important;
  color: var(--text) !important;
}
h3, h4, h5 {
  color: var(--text) !important;
  font-family: 'Mulish', sans-serif !important;
  font-weight: 800 !important;
}
p { color: var(--text-dim) !important; }

/* ── CINEMATIC HEADER ───────────────────────── */
.cinema-header-wrap {
  padding: 0.6rem 0 1.2rem 0;
}
.cinema-logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.4rem;
  letter-spacing: 7px;
  line-height: 1;
  background: linear-gradient(90deg, #ffffff 0%, #ff6b6b 40%, var(--gold) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.cinema-tagline {
  color: var(--muted);
  font-size: 0.76rem;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  margin-top: 2px;
  font-family: 'Mulish', sans-serif;
}

/* ── DIVIDER ────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1rem 0 !important;
}

/* ══════════════════════════════════════════════
   SEARCH INPUT
══════════════════════════════════════════════ */
.stTextInput > div > div > input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--text) !important;
  font-family: 'Mulish', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  padding: 0.75rem 1.2rem !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(229,9,20,0.15), var(--glow-red) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder {
  color: var(--muted) !important;
  font-style: italic !important;
}
.stTextInput label {
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-family: 'Mulish', sans-serif !important;
  font-weight: 700 !important;
}

/* ══════════════════════════════════════════════
   SELECTBOX (Search suggestions dropdown)
══════════════════════════════════════════════ */
.stSelectbox > div > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: 'Mulish', sans-serif !important;
}
.stSelectbox label {
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* ══════════════════════════════════════════════
   MOVIE CARD — POSTER + HOVER
══════════════════════════════════════════════ */
.movie-card-wrap {
  position: relative;
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--bg3);
  border: 1px solid var(--border);
  transition:
    transform 0.32s cubic-bezier(0.34, 1.56, 0.64, 1),
    box-shadow 0.32s ease,
    border-color 0.32s ease;
  cursor: pointer;
  aspect-ratio: 2/3;
  display: block;
}
.movie-card-wrap:hover {
  transform: translateY(-10px) scale(1.04);
  box-shadow: 0 24px 60px rgba(0,0,0,0.8), var(--glow-red);
  border-color: rgba(229,9,20,0.55);
}
.movie-card-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.45s ease;
}
.movie-card-wrap:hover img {
  transform: scale(1.08);
}
/* Gradient overlay shown on hover */
.card-hover-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to top,
    rgba(8,8,15,0.95) 0%,
    rgba(8,8,15,0.3) 40%,
    transparent 70%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
  display: flex;
  align-items: flex-end;
  padding: 10px;
}
.movie-card-wrap:hover .card-hover-overlay {
  opacity: 1;
}
.card-hover-title {
  font-family: 'Mulish', sans-serif;
  font-weight: 800;
  font-size: 0.78rem;
  color: #fff;
  line-height: 1.25;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
/* No-poster placeholder */
.no-poster-box {
  width: 100%;
  aspect-ratio: 2/3;
  background: linear-gradient(135deg, var(--bg4) 0%, var(--bg3) 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  font-size: 2.8rem;
  border-radius: var(--radius);
}
/* Card title below poster */
.movie-card-footer {
  font-family: 'Mulish', sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-dim);
  padding: 5px 2px 2px 2px;
  line-height: 1.25rem;
  height: 2.5rem;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

/* ══════════════════════════════════════════════
   OPEN BUTTON (in poster grid)
══════════════════════════════════════════════ */
.stButton > button {
  background: var(--accent-dim) !important;
  border: 1px solid var(--border-red) !important;
  color: #ff7070 !important;
  border-radius: 8px !important;
  font-family: 'Mulish', sans-serif !important;
  font-weight: 800 !important;
  font-size: 0.72rem !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  padding: 0.3rem 0.6rem !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
  margin-top: 4px !important;
}
.stButton > button:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  box-shadow: var(--glow-red) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

/* ══════════════════════════════════════════════
   SECTION HEADING BADGES
══════════════════════════════════════════════ */
.section-label {
  display: inline-block;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.55rem;
  letter-spacing: 3.5px;
  color: var(--text);
  border-bottom: 2px solid var(--accent);
  padding-bottom: 4px;
  margin-bottom: 1.1rem;
  margin-top: 0.5rem;
}
.section-label-gold {
  display: inline-block;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.55rem;
  letter-spacing: 3.5px;
  color: var(--text);
  border-bottom: 2px solid var(--gold);
  padding-bottom: 4px;
  margin-bottom: 1.1rem;
  margin-top: 0.5rem;
}
.feed-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: linear-gradient(90deg, var(--accent-dim), var(--gold-dim));
  border: 1px solid var(--border-red);
  border-radius: 20px;
  padding: 5px 16px;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text);
  margin-bottom: 1.2rem;
}

/* ══════════════════════════════════════════════
   DETAIL PAGE
══════════════════════════════════════════════ */
.detail-poster-wrap {
  border-radius: var(--radius-lg);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-lg), var(--glow-red);
  transition: box-shadow 0.35s ease;
}
.detail-poster-wrap:hover {
  box-shadow: var(--shadow-lg), 0 0 45px rgba(229,9,20,0.5);
}
.detail-poster-wrap img { width: 100%; display: block; }

.detail-info-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2rem 2.2rem;
  backdrop-filter: blur(12px);
  height: 100%;
}
.detail-movie-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.8rem;
  letter-spacing: 4px;
  line-height: 1;
  color: var(--text);
  margin-bottom: 0.6rem;
}
.detail-release-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--gold);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 0.85rem;
}
.genre-badge {
  display: inline-block;
  background: var(--accent-dim);
  border: 1px solid var(--border-red);
  color: #ff7070;
  border-radius: 16px;
  padding: 3px 12px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  margin-right: 6px;
  margin-bottom: 7px;
}
.detail-overview-label {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 1.3rem;
  margin-bottom: 0.55rem;
}
.detail-overview-text {
  color: var(--text-dim);
  font-size: 0.96rem;
  line-height: 1.8;
}
.back-btn .stButton > button {
  background: var(--bg4) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-dim) !important;
  font-size: 0.82rem !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-weight: 600 !important;
}
.back-btn .stButton > button:hover {
  background: var(--bg3) !important;
  border-color: var(--gold) !important;
  color: var(--gold) !important;
  box-shadow: var(--glow-gold) !important;
  transform: translateX(-4px) translateY(0) !important;
}

/* ── BACKDROP ───────────────────────────────── */
.backdrop-wrap {
  border-radius: var(--radius-lg);
  overflow: hidden;
  border: 1px solid var(--border);
  margin-top: 1.2rem;
  box-shadow: var(--shadow-lg);
}
.backdrop-wrap img { width: 100%; display: block; }

/* ══════════════════════════════════════════════
   ALERTS / INFO / WARNING
══════════════════════════════════════════════ */
.stAlert {
  border-radius: var(--radius) !important;
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
}

/* ══════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── SMALL MUTED (kept for backward compat) ─── */
.small-muted { color: var(--muted); font-size: 0.88rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING (single-file pages)
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=30)  # short cache for autocomplete
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


# ── REDESIGNED POSTER GRID ───────────────────────────────────────────────────
def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.markdown(
            "<div style='color:var(--muted);padding:1rem 0;font-family:Mulish,sans-serif;'>No movies to show.</div>",
            unsafe_allow_html=True,
        )
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols, gap="small")
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title   = m.get("title", "Untitled")
            poster  = m.get("poster_url")

            with colset[c]:
                # ── Card HTML (hover handled by CSS) ──
                if poster:
                    card_html = f"""
<div class="movie-card-wrap">
  <img src="{poster}" alt="{title}" loading="lazy" />
  <div class="card-hover-overlay">
    <div class="card-hover-title">{title}</div>
  </div>
</div>"""
                else:
                    card_html = f"""
<div class="movie-card-wrap">
  <div class="no-poster-box">🎬<br/>
    <span style="font-size:0.7rem;letter-spacing:1px;margin-top:6px;">NO POSTER</span>
  </div>
  <div class="card-hover-overlay">
    <div class="card-hover-title">{title}</div>
  </div>
</div>"""

                st.markdown(card_html, unsafe_allow_html=True)

                # ── Open Button ──
                if st.button("▶ Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                # ── Title below card ──
                st.markdown(
                    f"<div class='movie-card-footer'>{title}</div>",
                    unsafe_allow_html=True,
                )


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


# =============================
# IMPORTANT: Robust TMDB search parsing
# Supports BOTH API shapes:
# 1) raw TMDB: {"results":[{id,title,poster_path,...}]}
# 2) list cards: [{tmdb_id,title,poster_url,...}]
# =============================
def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    """
    Returns:
      suggestions: list[(label, tmdb_id)]
      cards: list[{tmdb_id,title,poster_url}]
    """
    keyword_l = keyword.strip().lower()

    # A) If API returns dict with 'results'
    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )

    # B) If API returns already as list
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            # might be {tmdb_id,title,poster_url}
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    # Word-match filtering (contains)
    matched = [x for x in raw_items if keyword_l in x["title"].lower()]

    # If nothing matched, fallback to raw list (so never blank)
    final_list = matched if matched else raw_items

    # Suggestions = top 10 labels
    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    # Cards = top N
    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("<div style='color:var(--muted);font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:1rem;'>Your Movie Universe</div>", unsafe_allow_html=True)

    if st.button("🏠  Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### 📡 Home Feed")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

    st.markdown("---")
    st.markdown(
        "<div style='color:var(--muted);font-size:0.68rem;letter-spacing:1px;text-transform:uppercase;text-align:center;padding-top:0.5rem;'>Powered by TMDB · NLP · FastAPI</div>",
        unsafe_allow_html=True,
    )

# =============================
# CINEMATIC HEADER
# =============================
st.markdown(
    """
<div class="cinema-header-wrap">
  <div class="cinema-logo">🎬 CineMatch</div>
  <div class="cinema-tagline">Search · Discover · Explore · Get Recommendations</div>
</div>
""",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Try: avenger, batman, love, inception..."
    )

    st.divider()

    # SEARCH MODE (Autocomplete + word-match results)
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                # Dropdown
                if suggestions:
                    labels = ["— Select a movie —"] + [s[0] for s in suggestions]
                    selected = st.selectbox("🎯 Suggestions", labels, index=0)

                    if selected != "— Select a movie —":
                        # map label -> id
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown(
                    f"<div class='section-label'>Search Results</div>",
                    unsafe_allow_html=True,
                )
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # HOME FEED MODE
    category_display = home_category.replace("_", " ").title()
    category_icon = {
        "trending": "🔥", "popular": "⭐", "top_rated": "🏆",
        "now_playing": "🎥", "upcoming": "📅"
    }.get(home_category, "🎬")

    st.markdown(
        f"<div class='feed-pill'>{category_icon} &nbsp; {category_display}</div>",
        unsafe_allow_html=True,
    )

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # ── Top Bar ──────────────────────────────────────────
    a, b = st.columns([4, 1])
    with a:
        st.markdown(
            "<div class='section-label'>📄 Movie Details</div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown("<div class='back-btn'>", unsafe_allow_html=True)
        if st.button("← Back to Home"):
            goto_home()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Fetch Details ─────────────────────────────────────
    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    # ── Layout: Poster LEFT, Details RIGHT ────────────────
    left, right = st.columns([1, 2.6], gap="large")

    with left:
        if data.get("poster_url"):
            st.markdown("<div class='detail-poster-wrap'>", unsafe_allow_html=True)
            st.image(data["poster_url"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='no-poster-box' style='height:420px;border-radius:18px;'>🎬</div>",
                unsafe_allow_html=True,
            )

    with right:
        release  = data.get("release_date") or "Unknown"
        genres   = data.get("genres", [])
        overview = data.get("overview") or "No overview available."

        # Genre badges HTML
        badges_html = "".join(
            f"<span class='genre-badge'>{g['name']}</span>" for g in genres
        ) or "<span class='genre-badge'>Unknown Genre</span>"

        st.markdown(
            f"""
<div class="detail-info-card">
  <div class="detail-movie-title">{data.get('title', '')}</div>
  <div class="detail-release-tag">📅 &nbsp; {release}</div>
  <div style="margin-bottom: 1rem;">{badges_html}</div>
  <div class="detail-overview-label">Overview</div>
  <div class="detail-overview-text">{overview}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Backdrop ──────────────────────────────────────────
    if data.get("backdrop_url"):
        st.markdown("<div class='backdrop-wrap'>", unsafe_allow_html=True)
        st.image(data["backdrop_url"], width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Recommendations ────────────────────────────────────
    st.markdown(
        "<div class='section-label'>✅ Recommendations</div>",
        unsafe_allow_html=True,
    )

    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown(
                "<div class='section-label'>🔎 Similar Movies (TF-IDF)</div>",
                unsafe_allow_html=True,
            )
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown(
                "<div class='section-label-gold'>🎭 More Like This (Genre)</div>",
                unsafe_allow_html=True,
            )
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=grid_cols, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")
