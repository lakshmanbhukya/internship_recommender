# QUICK REFERENCE CARD

## SETUP COMMANDS (Run Once)

```bash
# 1. Setup Kaggle credentials
mkdir %USERPROFILE%\.kaggle
# Place kaggle.json in C:\Users\laxman\.kaggle\

# 2. Install dependencies
pip install -r requirements-new.txt

# 3. Download dataset
python scripts/download_dataset.py

# 4. Preprocess data
python scripts/preprocess_data.py

# 5. Geocode cities
python models/geocode_cities.py

# 6. Generate embeddings (Colab)
# Upload notebooks/02_embedding_generation.ipynb to Colab
# Download internship_embeddings.npy and internship_metadata.csv
# Place in data/ folder

# 7. Create database
python database/create_database.py
```

## DAILY COMMANDS

```bash
# Start API
python api/main.py

# Test API
python test_api_v2.py

# Check health
curl http://localhost:8000/
```

## API ENDPOINTS

### Health Check
```bash
GET http://localhost:8000/
```

### Get Recommendations
```bash
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "skills": ["python", "machine learning"],
  "education": "B.Tech",
  "city": "Bangalore",
  "max_distance_km": 50,
  "min_stipend": 10000
}
```

## FILE LOCATIONS

```
data/raw/                              # Kaggle dataset
data/processed/internships_cleaned.csv # Cleaned data
data/internship_embeddings.npy         # Embeddings (35MB)
data/internship_metadata.csv          # Metadata
data/city_distance_matrix.json        # Distances
database/internships.db                # Main database (35MB)
```

## TROUBLESHOOTING

```bash
# Check Kaggle credentials
dir %USERPROFILE%\.kaggle\kaggle.json

# Check database
dir database\internships.db

# Check embeddings
dir data\internship_embeddings.npy

# Test imports
python -c "import fastapi; import sentence_transformers; print('OK')"

# Check port
netstat -ano | findstr :8000
```

## COMMON ERRORS

### "kaggle.json not found"
→ Place kaggle.json in C:\Users\laxman\.kaggle\

### "No module named 'sentence_transformers'"
→ pip install sentence-transformers torch

### "Database not found"
→ Run: python database/create_database.py

### "Embeddings file not found"
→ Run Colab notebook and download files

### "Port 8000 already in use"
→ Kill process: taskkill /F /PID <pid>

## VERIFICATION

```bash
# Check setup
python setup_v2.py

# Check files exist
dir data\internship_embeddings.npy
dir database\internships.db

# Test API
python test_api_v2.py
```

## DEPLOYMENT

```bash
# Build Docker image
docker build -t internship-recommender .

# Run container
docker run -p 8000:8000 internship-recommender

# Or deploy to Railway/Render
# Push to GitHub and connect repository
```

## USEFUL PATHS

```
Config:     config/settings.py
API:        api/main.py
Database:   database/internships.db
Scripts:    scripts/
Notebooks:  notebooks/
Docs:       START_HERE.md
```

## SUPPORT DOCS

- START_HERE.md - Quick start guide
- MIGRATION_GUIDE.md - Detailed migration
- IMPLEMENTATION_SUMMARY.md - Technical details
- STRUCTURE_COMPARISON.md - Old vs new

## QUICK TEST

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "skills": ["python"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 50,
        "min_stipend": 0
    }
)

print(response.json())
```

## PERFORMANCE TARGETS

- API startup: <5 seconds
- Search time: <100ms
- Database size: ~35MB
- Memory usage: ~500MB

## NEXT STEPS

1. Get Kaggle credentials
2. Run setup commands
3. Generate embeddings on Colab
4. Create database
5. Start API
6. Test
7. Deploy

---

**Need help? Check START_HERE.md**
