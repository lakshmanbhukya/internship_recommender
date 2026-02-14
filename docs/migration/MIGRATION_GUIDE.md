# Internship Recommender System v2.0 - Industry Grade

## 🚀 What's New in v2.0

- **SQLite-vec**: Lightweight vector database (~35MB vs ChromaDB)
- **Production-ready API**: Clean architecture with proper separation
- **Kaggle Integration**: Direct dataset download pipeline
- **GPU Embeddings**: Colab T4 notebook for fast embedding generation
- **Better Performance**: Optimized search and ranking

## 📋 Migration Steps

### Step 1: Get Kaggle API Credentials
1. Go to https://www.kaggle.com/settings
2. Create API token → download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json`
4. Windows: `C:\Users\<username>\.kaggle\kaggle.json`

### Step 2: Download & Preprocess Data
```bash
# Install new requirements
pip install -r requirements-new.txt

# Download dataset from Kaggle
python scripts/download_dataset.py

# Preprocess data
python scripts/preprocess_data.py

# Geocode cities
python models/geocode_cities.py
```

### Step 3: Generate Embeddings (Google Colab)
1. Open `notebooks/02_embedding_generation.ipynb` in Colab
2. Set runtime to **GPU (T4)**
3. Upload `data/processed/internships_cleaned.csv`
4. Run all cells
5. Download `internship_embeddings.npy` and `internship_metadata.csv`
6. Place both files in `data/` folder

### Step 4: Create Database
```bash
python database/create_database.py
```

### Step 5: Run New API
```bash
# Start server
python api/main.py

# Or with uvicorn
uvicorn api.main:app --reload
```

### Step 6: Test API
```bash
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

## 📁 New Structure

```
internship_recommender/
├── api/                    # Production API
│   ├── main.py            # FastAPI app
│   ├── config.py          # Settings
│   ├── schemas.py         # Pydantic models
│   ├── database.py        # SQLite operations
│   ├── recommendations.py # Engine
│   └── utils.py           # Helpers
├── config/                # Configuration
│   └── settings.py        # Mappings & paths
├── database/              # SQLite database
│   ├── create_database.py
│   └── internships.db     # 35MB vector DB
├── data/                  # Data files
│   ├── raw/               # Kaggle dataset
│   ├── processed/         # Cleaned data
│   ├── internship_embeddings.npy
│   ├── internship_metadata.csv
│   ├── geocoding_cache.json
│   └── city_distance_matrix.json
├── models/                # ML models
│   └── geocode_cities.py
├── scripts/               # Data pipeline
│   ├── download_dataset.py
│   └── preprocess_data.py
├── notebooks/             # Colab notebooks
│   └── 02_embedding_generation.ipynb
├── Dockerfile
└── requirements-new.txt
```

## 🗑️ Files to Delete (After Migration)

```bash
# Old ChromaDB system
rm -rf chroma/
rm -rf vector_store/
rm -rf embeddings/
rm -rf semantic_recommender/
rm -rf resume_parser/
rm -rf utils/

# Old files
rm connection.py
rm recommender.py
rm main.py
rm migrate_data.py
rm setup_semantic.py
rm test_semantic.py
rm tfidf_vectorizer.joblib
```

## 🎯 API Endpoints

### Health Check
```
GET /
```

### Get Recommendations
```
POST /recommend
{
  "skills": ["python", "data science"],
  "education": "B.Tech",
  "city": "Bangalore",
  "max_distance_km": 50,
  "min_stipend": 10000,
  "preferred_sectors": ["technology"]
}
```

## 🚢 Deployment

### Docker
```bash
docker build -t internship-recommender .
docker run -p 8000:8000 internship-recommender
```

### Railway/Render
1. Push to GitHub
2. Connect repository
3. Set build command: `pip install -r requirements-new.txt`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Upload `database/internships.db` as persistent volume

## 📊 Performance

- **Database Size**: 35MB (vs 200MB+ ChromaDB)
- **Search Speed**: <100ms for 10 results
- **Memory Usage**: ~500MB (model loaded)
- **Startup Time**: ~5 seconds

## 🔧 Configuration

Edit `.env`:
```
DATABASE_PATH=database/internships.db
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
DEFAULT_TOP_K=10
MAX_DISTANCE_KM=100
```

## ✅ Migration Checklist

- [ ] Kaggle credentials configured
- [ ] Dataset downloaded
- [ ] Data preprocessed
- [ ] Cities geocoded
- [ ] Embeddings generated on Colab
- [ ] Database created
- [ ] New API tested
- [ ] Old files deleted
- [ ] Deployed to production

## 🆘 Troubleshooting

**Issue**: Kaggle download fails
- Solution: Check `~/.kaggle/kaggle.json` exists and has correct permissions

**Issue**: Colab out of memory
- Solution: Reduce batch_size to 32 in embedding generation

**Issue**: sqlite-vec not loading
- Solution: Database will work without it (fallback mode)

**Issue**: Model download slow
- Solution: First run downloads ~2GB model, subsequent runs use cache

## 📞 Support

For issues, check logs or raise an issue on GitHub.
