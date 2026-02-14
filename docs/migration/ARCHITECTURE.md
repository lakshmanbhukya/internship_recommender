# SYSTEM ARCHITECTURE DIAGRAM

## DATA FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

1. KAGGLE DATASET
   ↓
   [scripts/download_dataset.py]
   ↓
   data/raw/internship_opportunities_2025.csv (8,485 records)

2. PREPROCESSING
   ↓
   [scripts/preprocess_data.py]
   ↓
   data/processed/internships_cleaned.csv
   - Normalized cities
   - Parsed skills
   - Cleaned education
   - Freshness scores

3. GEOCODING
   ↓
   [models/geocode_cities.py]
   ↓
   data/geocoding_cache.json
   data/city_distance_matrix.json

4. EMBEDDING GENERATION (Colab T4 GPU)
   ↓
   [notebooks/02_embedding_generation.ipynb]
   ↓
   data/internship_embeddings.npy (35MB)
   data/internship_metadata.csv

5. DATABASE CREATION
   ↓
   [database/create_database.py]
   ↓
   database/internships.db (35MB SQLite)
```

## API ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                                │
└─────────────────────────────────────────────────────────────────┘

CLIENT REQUEST
   ↓
   POST /recommend
   {
     "skills": ["python", "ML"],
     "education": "B.Tech",
     "city": "Bangalore",
     "max_distance_km": 50,
     "min_stipend": 10000
   }
   ↓
┌──────────────────────┐
│   api/main.py        │  FastAPI Application
│   - Route handling   │
│   - CORS middleware  │
│   - Error handling   │
└──────────────────────┘
   ↓
┌──────────────────────┐
│  api/schemas.py      │  Pydantic Validation
│  - UserProfile       │
│  - InternshipResponse│
└──────────────────────┘
   ↓
┌──────────────────────┐
│ api/recommendations  │  Recommendation Engine
│ .py                  │
│ - Load model         │
│ - Generate embedding │
│ - Search database    │
│ - Calculate scores   │
└──────────────────────┘
   ↓
┌──────────────────────┐
│  api/database.py     │  Database Layer
│  - SQLite operations │
│  - Vector search     │
│  - Filtering         │
└──────────────────────┘
   ↓
┌──────────────────────┐
│  api/utils.py        │  Utilities
│  - Distance calc     │
│  - Score calculation │
└──────────────────────┘
   ↓
┌──────────────────────┐
│ database/            │  SQLite Database
│ internships.db       │
│ - Metadata           │
│ - Embeddings (BLOB)  │
│ - Indexes            │
└──────────────────────┘
   ↓
RESPONSE
{
  "total_results": 10,
  "recommendations": [
    {
      "id": "...",
      "role": "ML Engineer",
      "company": "TechCorp",
      "match_score": 87.5,
      "distance_km": 12.3,
      ...
    }
  ]
}
```

## COMPONENT INTERACTION

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM COMPONENTS                             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Client     │
│  (Browser/   │
│   Postman)   │
└──────┬───────┘
       │ HTTP POST
       ↓
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Server                           │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐          │
│  │   Routes   │→ │ Validation  │→ │   Engine     │          │
│  │ (main.py)  │  │ (schemas.py)│  │(recommend.py)│          │
│  └────────────┘  └─────────────┘  └──────┬───────┘          │
│                                            │                   │
│  ┌────────────────────────────────────────┘                   │
│  │                                                             │
│  ↓                                                             │
│  ┌──────────────┐         ┌─────────────────┐                │
│  │  Sentence    │         │   Database      │                │
│  │ Transformers │         │   Operations    │                │
│  │   (Model)    │         │  (database.py)  │                │
│  └──────────────┘         └────────┬────────┘                │
│                                     │                          │
└─────────────────────────────────────┼──────────────────────────┘
                                      │
                                      ↓
                            ┌──────────────────┐
                            │  SQLite Database │
                            │  internships.db  │
                            │                  │
                            │  - Metadata      │
                            │  - Embeddings    │
                            │  - Indexes       │
                            └──────────────────┘
```

## SEARCH ALGORITHM

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH FLOW                                   │
└─────────────────────────────────────────────────────────────────┘

USER QUERY
   ↓
1. CREATE QUERY TEXT
   "Skills: python, ML
    Education: B.Tech
    Location: Bangalore"
   ↓
2. GENERATE EMBEDDING
   [sentence-transformers]
   → 1024-dim vector
   ↓
3. DATABASE SEARCH
   [SQLite query]
   - Filter by education
   - Filter by stipend
   - Get top 50 candidates
   ↓
4. CALCULATE DISTANCES
   [geopy + distance matrix]
   - User city → Internship city
   - Filter by max_distance
   ↓
5. VECTOR SIMILARITY
   [numpy]
   - Cosine similarity
   - Between query & internship embeddings
   ↓
6. HYBRID SCORING
   [utils.py]
   Score = Semantic(50%) + Freshness(30%) + Distance(20%)
   ↓
7. RANK & RETURN
   - Sort by match_score
   - Return top 10
   ↓
RESULTS
```

## SCORING FORMULA

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATCH SCORE (0-100)                           │
└─────────────────────────────────────────────────────────────────┘

SEMANTIC SIMILARITY (50 points)
   = (1 - vector_distance) × 50
   
FRESHNESS BOOST (30 points)
   = freshness_score × 30
   (1.0 for today, decays over 30 days)
   
DISTANCE SCORE (20 points)
   = (1 - distance_km / max_distance) × 20
   (if within range, else 0)

FINAL SCORE
   = Semantic + Freshness + Distance
   = 0 to 100
```

## DEPLOYMENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT OPTIONS                            │
└─────────────────────────────────────────────────────────────────┘

OPTION 1: Docker Container
┌──────────────────────────────────────┐
│  Docker Container                     │
│  ┌────────────────────────────────┐  │
│  │  FastAPI App                   │  │
│  │  + SQLite DB (35MB)            │  │
│  │  + Model cache                 │  │
│  └────────────────────────────────┘  │
│  Port: 8000                           │
└──────────────────────────────────────┘

OPTION 2: Cloud Platform (Railway/Render)
┌──────────────────────────────────────┐
│  Cloud Instance                       │
│  ┌────────────────────────────────┐  │
│  │  Python Runtime                │  │
│  │  + Dependencies                │  │
│  │  + SQLite DB (persistent vol) │  │
│  └────────────────────────────────┘  │
│  Auto-scaling enabled                 │
└──────────────────────────────────────┘

OPTION 3: Local Development
┌──────────────────────────────────────┐
│  Local Machine                        │
│  python api/main.py                   │
│  http://localhost:8000                │
└──────────────────────────────────────┘
```

## FILE STRUCTURE VISUAL

```
internship_recommender/
│
├── api/                    ← Production API
│   ├── main.py            ← Entry point
│   ├── config.py          ← Settings
│   ├── schemas.py         ← Models
│   ├── database.py        ← DB ops
│   ├── recommendations.py ← Engine
│   └── utils.py           ← Helpers
│
├── config/                 ← Configuration
│   └── settings.py        ← Mappings
│
├── database/               ← SQLite DB
│   ├── create_database.py ← Creator
│   └── internships.db     ← 35MB DB ★
│
├── data/                   ← Data files
│   ├── raw/               ← Kaggle
│   ├── processed/         ← Cleaned
│   ├── *.npy              ← Embeddings ★
│   └── *.json             ← Caches
│
├── models/                 ← ML models
│   └── geocode_cities.py  ← Geocoding
│
├── scripts/                ← Pipeline
│   ├── download_dataset.py
│   └── preprocess_data.py
│
├── notebooks/              ← Colab
│   └── 02_embedding_*.ipynb
│
├── Dockerfile              ← Container
├── requirements-new.txt    ← Deps
└── *.md                    ← Docs

★ = Critical files (generated)
```

## TECHNOLOGY STACK

```
┌─────────────────────────────────────────────────────────────────┐
│                    TECH STACK                                    │
└─────────────────────────────────────────────────────────────────┘

BACKEND
   FastAPI          → Web framework
   Uvicorn          → ASGI server
   Pydantic         → Validation

MACHINE LEARNING
   sentence-transformers → Embeddings
   BAAI/bge-m3          → Model
   PyTorch              → ML backend
   NumPy                → Computations

DATABASE
   SQLite3          → Database
   sqlite-vec       → Vector extension (optional)

DATA PROCESSING
   Pandas           → Data manipulation
   scikit-learn     → ML utilities
   geopy            → Geocoding

DEPLOYMENT
   Docker           → Containerization
   Railway/Render   → Cloud hosting
```

---

**Visual guide complete!**
**See START_HERE.md to begin implementation.**
