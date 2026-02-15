<div align="center">

# 🎯 Internship Recommender System

**AI-powered internship recommendation engine built with semantic search, hybrid retrieval, and intelligent matching.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [How It Works](#-how-it-works) · [Performance](#-performance)

</div>

---

## 📌 Overview

A recommendation system that matches students to relevant internships from **8,483 real internship listings** scraped from Internshala. Unlike basic keyword search, this system uses **semantic understanding** via BGE-M3 embeddings + FAISS vector search, combined with **lexical matching** via SQLite FTS5, fused through **Reciprocal Rank Fusion (RRF)** — the same technique used by enterprise search engines.

### Dataset

| Metric | Value |
|--------|-------|
| Total Internships | **8,483** |
| Unique Roles | **1,977** |
| Companies | **5,114** |
| Cities | **75** |
| Source | [Internship Opportunities in India (2025)](https://www.kaggle.com/datasets) — Internshala |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | FAISS semantic search + FTS5 BM25 lexical search with RRF fusion |
| **Dual Search Modes** | Full mode (GPU/CPU with BGE-M3) or Lightweight mode (512 MB, no model) |
| **Skill Matching** | Direct skill overlap scoring + synonym expansion for 25+ skill domains |
| **Education Hierarchy** | B.Tech → Diploma → Any — users see all internships they qualify for |
| **Location-Aware** | Distance-based filtering using a pre-computed city distance matrix (75 cities) |
| **Stipend Filtering** | Filter by minimum expected stipend |
| **Seniority Guard** | Automatically deprioritizes senior roles for student users |
| **Fast API** | RESTful API with health checks, CORS, and structured responses |
| **Docker Ready** | Multi-stage Dockerfile, Render deployment config included |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│                   POST /recommend                    │
└───────────────┬─────────────────────┬───────────────┘
                │                     │
        ┌───────▼───────┐     ┌───────▼───────┐
        │  Full Mode     │     │ Lightweight    │
        │  (BGE-M3 +     │     │ (Keyword +     │
        │   FAISS +      │     │  Synonyms +    │
        │   FTS5)        │     │  Filters)      │
        └───────┬───────┘     └───────┬───────┘
                │                     │
    ┌───────────▼─────────┐           │
    │  Reciprocal Rank    │           │
    │  Fusion (RRF)       │           │
    │  60% Semantic       │           │
    │  40% Lexical        │           │
    └───────────┬─────────┘           │
                │                     │
        ┌───────▼─────────────────────▼───────┐
        │         Filter & Score Pipeline       │
        │                                       │
        │  ├─ Education Hierarchy Filter        │
        │  ├─ Stipend Threshold Filter          │
        │  ├─ City Distance Filter              │
        │  ├─ Seniority Penalty                 │
        │  └─ Direct Skill Overlap Bonus        │
        └───────────────────┬─────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Top-K Results │
                    │  (Scored 0-100)│
                    └───────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **Embeddings** | BAAI/bge-m3 (1024-dim) via sentence-transformers |
| **Vector Search** | FAISS HNSW index (efSearch=64) |
| **Lexical Search** | SQLite FTS5 (BM25) |
| **Database** | SQLite3 (~40 MB) |
| **Data Processing** | Pandas, NumPy |
| **Geospatial** | Pre-computed distance matrix (geopy) |
| **Deployment** | Docker, Render |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~2.5 GB RAM (Full mode) or ~512 MB (Lightweight mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/lakshmanbhukya/internship_recommender.git
cd internship_recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-new.txt
```

### Run the API

```bash
# Lightweight mode (fast, low memory — default for deployment)
set LIGHTWEIGHT_MODE=true      # Windows
export LIGHTWEIGHT_MODE=true   # Linux/Mac
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Full mode (semantic search with BGE-M3)
set LIGHTWEIGHT_MODE=false
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
python scripts/test_recommendations.py
```

---

## 📡 API Reference

### `GET /` — Health Check

```json
{
  "status": "healthy",
  "database_connected": true,
  "model_loaded": true,
  "total_internships": 8483,
  "version": "2.1.0"
}
```

### `POST /recommend` — Get Recommendations

**Request:**
```json
{
  "skills": ["Python", "Django", "REST API"],
  "education": "B.Tech",
  "city": "Bangalore",
  "max_distance_km": 100,
  "min_stipend": 5000
}
```

**Response:**
```json
{
  "query": { ... },
  "total_results": 10,
  "recommendations": [
    {
      "id": "2025BLR0042",
      "role": "Backend Development",
      "company": "TechCorp",
      "city": "Bangalore",
      "stipend_min": 10000,
      "stipend_max": 15000,
      "duration_months": 3,
      "education_req": "B.Tech",
      "skills": ["Python", "Django", "REST API", "PostgreSQL"],
      "match_score": 87.3,
      "distance_km": 0.0,
      "freshness_score": 0.3
    }
  ],
  "metadata": {
    "version": "2.1.0",
    "model": "BAAI/bge-m3"
  }
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `skills` | `string[]` | ✅ | — | 1–20 skills |
| `education` | `string` | ✅ | — | `B.Tech`, `B.Sc`, `B.Com`, `B.A`, `Diploma`, `Any` |
| `city` | `string` | ✅ | — | Indian city name or `Remote` |
| `max_distance_km` | `int` | ❌ | `50` | 0–500 km radius |
| `min_stipend` | `int` | ❌ | `0` | Minimum monthly stipend (₹) |

---

## ⚙ How It Works

### 1. Data Pipeline

```
Raw CSVs (9 city files)
    │
    ▼
preprocess_data.py ─── normalize cities, parse stipends, clean skills
    │
    ▼
create_database.py ─── SQLite DB + FTS5 full-text index
    │
    ▼
build_faiss_index.py ── FAISS HNSW index from BGE-M3 embeddings
```

### 2. Search Flow (Full Mode)

1. **Encode Query** — User's skills + education + city → BGE-M3 → 1024-dim vector
2. **Semantic Retrieval** — FAISS nearest-neighbor search → top 50 candidates
3. **Lexical Retrieval** — FTS5 BM25 full-text search on skills + profile → top 50 candidates
4. **RRF Fusion** — Merge both result sets using Reciprocal Rank Fusion (60/40 weight)
5. **Filter & Score** — Apply education hierarchy, stipend threshold, city distance, seniority penalty, and direct skill overlap bonus
6. **Return Top-K** — Sorted by final match score (0–100)

### 3. Scoring Formula

```
final_score = base_relevance × distance_factor × seniority_factor × skill_bonus × 100

Where:
  base_relevance  = Normalized RRF fusion score (0–1)
  distance_factor = max(0.3, 1 - distance/max_distance)
  seniority_factor = 0.6 if senior role, else 1.0
  skill_bonus     = 1.0 + (overlap_count / user_skills) × 0.4
```

### 4. Education Hierarchy

Users see internships at or below their education level:

```
PhD > M.Tech = MBA > B.Tech = B.Sc = B.Com = B.A > Diploma > Any
```

A `B.Tech` student sees `B.Tech`, `Diploma`, and `Any` internships — but not `M.Tech`.

---

## 📊 Performance

### Accuracy (Test Suite)

| Test Case | Skills | Relevance |
|-----------|--------|-----------|
| Backend Developer | Python, Django, REST API | ✅ 100% |
| Frontend Developer | React, JavaScript, HTML, CSS | ✅ 100% |
| Data Science | Python, Machine Learning, Pandas | ✅ 100% |
| Marketing | Social Media, Content Writing | ✅ 100% |
| **Overall** | | **100%** |

### Latency

| Mode | Avg Latency | Memory |
|------|-------------|--------|
| Lightweight | **36 ms** | 512 MB |
| Full (BGE-M3) | ~400 ms | 2.3 GB |

### Edge Cases Handled

- ✅ Single skill queries
- ✅ 10+ multi-skill queries
- ✅ Unknown cities (graceful fallback)
- ✅ Zero-distance (same city only)
- ✅ High stipend filters
- ✅ Remote work matching

---

## 📁 Project Structure

```
internship_recommender/
├── api/
│   ├── main.py                 # FastAPI app, endpoints, startup
│   ├── hybrid_search.py        # FAISS + FTS5 + RRF fusion engine
│   ├── lightweight_search.py   # Keyword + synonym matching engine
│   ├── engine_selector.py      # Mode selector (env-based)
│   ├── schemas.py              # Pydantic request/response models
│   ├── config.py               # API settings (pydantic-settings)
│   └── utils.py                # Education hierarchy, city distance, date parsing
├── config/
│   └── settings.py             # Paths, model config, city/education mappings
├── data/
│   ├── raw/                    # 9 city CSVs + merged dataset
│   ├── processed/              # Cleaned + enhanced CSVs
│   ├── faiss_index.bin         # FAISS HNSW index (~35 MB)
│   ├── embeddings_v2.npy       # BGE-M3 embeddings (~33 MB)
│   ├── id_mapping.json         # FAISS index → internship ID map
│   └── city_distance_matrix.json
├── database/
│   ├── create_database.py      # DB creation + FTS5 indexing
│   └── internships.db          # SQLite database (~40 MB)
├── scripts/
│   ├── preprocess_data.py      # Data cleaning pipeline
│   ├── build_faiss_index.py    # FAISS index builder
│   ├── update_embeddings.py    # Embedding update pipeline
│   ├── test_recommendations.py # Test suite (5 test categories)
│   └── ...
├── notebooks/                  # Colab notebooks for GPU embedding generation
├── docs/                       # Additional documentation
├── Dockerfile                  # Multi-stage Docker build
├── render.yaml                 # Render deployment config
└── requirements-new.txt        # Python dependencies
```

---

## 🐳 Deployment

### Docker

```bash
docker build -t internship-recommender .
docker run -p 8000:8000 internship-recommender
```

### Render

The `render.yaml` is pre-configured. Connect your GitHub repo to [Render](https://render.com) and deploy.

---

## 🧪 Testing

```bash
# Run the full test suite
python scripts/test_recommendations.py
```

**Test categories:**
1. **Lightweight Mode** — 5 persona-based searches
2. **Edge Cases** — Single skill, many skills, unknown city, high stipend, zero distance
3. **Accuracy** — Keyword relevance scoring against expected results
4. **Performance** — Latency benchmarks across result sizes
5. **Health Check** — API readiness verification

---

## 🔮 Future Improvements

- [ ] Frontend UI (React/Next.js dashboard)
- [ ] User feedback loop for relevance tuning
- [ ] Real-time data ingestion from Internshala API
- [ ] Fine-tuned embeddings on internship-specific corpus
- [ ] A/B testing framework for scoring weight optimization
- [ ] Redis caching for repeated queries

---

## 📄 License

MIT

---

<div align="center">

**Built with ❤️ by [Lakshman Bhukya](https://github.com/lakshmanbhukya)**

</div>
