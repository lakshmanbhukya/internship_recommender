# Internship Recommender System v2.0

Industry-grade internship recommendation system with dual search modes: Lightweight (fast) and BGE-M3 (accurate).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-new.txt

# 2. Run API (Lightweight mode - default)
python api/main.py

# 3. Test both modes
python test_search_modes.py both
```

## Search Modes

### Lightweight Mode (Default)
- **Accuracy**: 62.5%
- **Speed**: 16ms per query
- **Memory**: 512 MB
- **Best for**: High traffic, marketing roles

### BGE-M3 Full Mode
- **Accuracy**: 82%
- **Speed**: 2000ms per query
- **Memory**: 2.3 GB
- **Best for**: Technical roles, accuracy-critical

### Switch Modes
```bash
# Lightweight (fast)
export LIGHTWEIGHT_MODE=true
python api/main.py

# BGE-M3 (accurate)
export LIGHTWEIGHT_MODE=false
python api/main.py
```

## Features

- **Dual Search Modes**: Choose speed or accuracy
- **Semantic Search**: BGE-M3 embeddings (1024-dim)
- **Hybrid Scoring**: 70% semantic + 30% keyword
- **Location Filtering**: Distance-based matching
- **Fast Performance**: <100ms in lightweight mode
- **Production-Ready**: Clean FastAPI architecture

## API Usage

```bash
# Health check
curl http://localhost:8000/

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["python", "machine learning"],
    "education": "B.Tech",
    "city": "Bangalore",
    "max_distance_km": 50,
    "min_stipend": 10000
  }'
```

## Project Structure

```
api/                    - FastAPI application
config/                 - Configuration & settings
database/               - SQLite database (35MB)
data/                   - Datasets & embeddings
models/                 - ML models & geocoding
scripts/                - Data pipeline scripts
notebooks/              - Colab notebooks
docs/migration/         - Migration documentation
```

## Documentation

- **[Search Modes Comparison](SEARCH_MODES_COMPARISON.md)** - Comprehensive comparison of Lightweight vs BGE-M3
- **[API Reference](docs/migration/QUICK_REFERENCE.md)** - API documentation
- **[Architecture](docs/migration/ARCHITECTURE.md)** - System architecture

## Tech Stack

- FastAPI + Uvicorn
- sentence-transformers (BGE-M3)
- SQLite3 + FAISS
- Pandas + NumPy + geopy

## Performance Comparison

| Mode | Accuracy | Latency | Memory | Cost/1K queries |
|------|----------|---------|--------|----------------|
| Lightweight | 62.5% | 16ms | 512 MB | $0.001 |
| BGE-M3 | 82% | 2000ms | 2.3 GB | $1.00 |
| Hybrid | 88% | 400ms | 1.5 GB | $0.04 |

## Testing

```bash
# Test lightweight mode
python test_search_modes.py lightweight

# Test BGE-M3 mode
python test_search_modes.py bge-m3

# Compare both
python test_search_modes.py both
```

## Deployment

```bash
# Docker
docker build -t internship-recommender .
docker run -p 8000:8000 internship-recommender

# Or deploy to Railway/Render
# See docs/migration/MIGRATION_GUIDE.md
```

## License

MIT
