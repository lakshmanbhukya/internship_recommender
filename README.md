# Internship Recommender System v2.0

Industry-grade internship recommendation system using semantic search with SQLite vector database.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-new.txt

# 2. Setup Kaggle credentials (get from https://www.kaggle.com/settings)
# Place kaggle.json in C:\Users\<username>\.kaggle\

# 3. Run data pipeline
python scripts/download_dataset.py
python scripts/preprocess_data.py
python models/geocode_cities.py

# 4. Generate embeddings on Google Colab (GPU T4)
# Upload notebooks/02_embedding_generation.ipynb
# Download files to data/ folder

# 5. Create database
python database/create_database.py

# 6. Start API
python api/main.py
```

## Features

- **Semantic Search**: BGE-M3 embeddings (1024-dim) for contextual matching
- **Lightweight**: 35MB SQLite database vs 200MB+ alternatives
- **Fast**: <100ms search, <5s startup
- **Production-Ready**: Clean FastAPI architecture
- **Automated Pipeline**: Kaggle → Preprocessing → Embeddings → Database

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

- **Migration Guide**: [docs/migration/START_HERE.md](docs/migration/START_HERE.md)
- **API Reference**: [docs/migration/QUICK_REFERENCE.md](docs/migration/QUICK_REFERENCE.md)
- **Architecture**: [docs/migration/ARCHITECTURE.md](docs/migration/ARCHITECTURE.md)
- **Old System**: [docs/old_system/README_OLD.md](docs/old_system/README_OLD.md)

## Tech Stack

- FastAPI + Uvicorn
- sentence-transformers (BGE-M3)
- SQLite3 + NumPy
- Pandas + geopy
- Kaggle API

## Performance

- Database: 35MB
- Startup: <5 seconds
- Search: <100ms
- Memory: ~500MB
- Records: 8,485 internships

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
