# COMPLETE IMPLEMENTATION GUIDE

## CURRENT STATUS

✅ All new files created successfully!
✅ Directory structure ready
✅ Scripts and API code ready
⚠️ Kaggle credentials needed
⚠️ sentence-transformers needs installation

---

## WHAT WAS CREATED

### New Files (20+ files):
```
config/settings.py                    - Configuration & mappings
scripts/download_dataset.py           - Kaggle downloader
scripts/preprocess_data.py            - Data cleaning
models/geocode_cities.py              - City geocoding
notebooks/02_embedding_generation.ipynb - Colab notebook
database/create_database.py           - SQLite DB creator
api/config.py                         - API settings
api/schemas.py                        - Pydantic models
api/utils.py                          - Helper functions
api/database.py                       - DB operations
api/recommendations.py                - Recommendation engine
api/main.py                           - FastAPI app
requirements-new.txt                  - Dependencies
Dockerfile                            - Container config
setup_v2.py                           - Setup script
test_api_v2.py                        - Test script
MIGRATION_GUIDE.md                    - Migration docs
IMPLEMENTATION_SUMMARY.md             - Summary
```

---

## STEP-BY-STEP EXECUTION

### STEP 1: Get Kaggle Credentials (DO THIS NOW!)

1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Place it here: `C:\Users\laxman\.kaggle\kaggle.json`

```cmd
mkdir %USERPROFILE%\.kaggle
move Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### STEP 2: Install Dependencies

```cmd
pip install -r requirements-new.txt
```

This installs:
- fastapi, uvicorn
- sentence-transformers, torch
- pandas, numpy, scikit-learn
- geopy, kaggle

### STEP 3: Download Dataset

```cmd
python scripts/download_dataset.py
```

Expected output:
- File: `data/raw/internship_opportunities_2025.csv` (~2-3 MB)
- ~8,485 internship records

### STEP 4: Preprocess Data

```cmd
python scripts/preprocess_data.py
```

Expected output:
- File: `data/processed/internships_cleaned.csv` (~2 MB)
- Normalized cities, skills, education
- Freshness scores calculated

### STEP 5: Geocode Cities

```cmd
python models/geocode_cities.py
```

Expected output:
- File: `data/geocoding_cache.json`
- File: `data/city_distance_matrix.json`
- ~12 cities geocoded

### STEP 6: Generate Embeddings (COLAB - REQUIRES YOUR ACTION!)

1. Open Google Colab: https://colab.research.google.com/
2. Upload `notebooks/02_embedding_generation.ipynb`
3. Change runtime: Runtime → Change runtime type → GPU (T4)
4. Run all cells:
   - Upload `data/processed/internships_cleaned.csv` when prompted
   - Wait ~3-4 minutes for embeddings
   - Download 2 files:
     * `internship_embeddings.npy` (~35 MB)
     * `internship_metadata.csv` (~2 MB)
5. Place both files in `data/` folder

### STEP 7: Create Database

```cmd
python database/create_database.py
```

Expected output:
- File: `database/internships.db` (~35 MB)
- 8,485 records with embeddings

### STEP 8: Run API

```cmd
python api/main.py
```

Expected output:
```
Loading model: BAAI/bge-m3
Model loaded on cpu
Database connected
Ready to serve recommendations!
```

### STEP 9: Test API

Open new terminal:

```cmd
python test_api_v2.py
```

Or test with curl:

```cmd
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"skills\":[\"python\",\"machine learning\"],\"education\":\"B.Tech\",\"city\":\"Bangalore\",\"max_distance_km\":50,\"min_stipend\":10000}"
```

---

## WHAT TO DELETE (After Migration Complete)

```cmd
# Old ChromaDB system
rmdir /s /q chroma
rmdir /s /q vector_store
rmdir /s /q embeddings
rmdir /s /q semantic_recommender
rmdir /s /q resume_parser
rmdir /s /q utils

# Old files
del connection.py
del recommender.py
del main.py
del migrate_data.py
del setup_semantic.py
del test_semantic.py
del tfidf_vectorizer.joblib
del requirements.txt
```

Keep:
- `.env` (update if needed)
- `.gitignore`
- `README.md` (update with new instructions)

---

## TROUBLESHOOTING

### Issue: Kaggle download fails
```
Error: Could not find kaggle.json
Solution: Check C:\Users\laxman\.kaggle\kaggle.json exists
```

### Issue: sentence-transformers not found
```
Solution: pip install sentence-transformers torch
```

### Issue: Colab out of memory
```
Solution: In notebook, change batch_size=64 to batch_size=32
```

### Issue: Database creation fails
```
Error: File not found
Solution: Make sure internship_embeddings.npy and internship_metadata.csv are in data/
```

### Issue: API won't start
```
Error: Database not found
Solution: Run python database/create_database.py first
```

---

## VERIFICATION CHECKLIST

Before proceeding to next step, verify:

- [ ] Step 1: kaggle.json exists in C:\Users\laxman\.kaggle\
- [ ] Step 2: All packages installed (no import errors)
- [ ] Step 3: data/raw/*.csv exists (~2-3 MB)
- [ ] Step 4: data/processed/internships_cleaned.csv exists
- [ ] Step 5: data/city_distance_matrix.json exists
- [ ] Step 6: data/internship_embeddings.npy exists (~35 MB)
- [ ] Step 6: data/internship_metadata.csv exists
- [ ] Step 7: database/internships.db exists (~35 MB)
- [ ] Step 8: API starts without errors
- [ ] Step 9: Test returns recommendations

---

## EXPECTED TIMELINE

- Step 1: 5 minutes (Kaggle setup)
- Step 2: 10 minutes (pip install)
- Step 3: 2 minutes (download)
- Step 4: 1 minute (preprocess)
- Step 5: 2 minutes (geocode)
- Step 6: 5 minutes (Colab setup + run)
- Step 7: 2 minutes (database)
- Step 8: 30 seconds (API start)
- Step 9: 10 seconds (test)

**Total: ~30 minutes**

---

## NEXT IMMEDIATE ACTION

1. Get Kaggle credentials (Step 1)
2. Run: `pip install -r requirements-new.txt`
3. Continue with Step 3

---

## SUPPORT

If stuck, check:
1. MIGRATION_GUIDE.md - Detailed migration docs
2. IMPLEMENTATION_SUMMARY.md - Technical summary
3. Error messages in terminal

---

## SUCCESS INDICATORS

✅ API responds at http://localhost:8000/
✅ Health check shows database_connected: true
✅ Recommendations return in <100ms
✅ Match scores between 50-100
✅ Distance filtering works
✅ Education filtering works

---

Ready to start? Run:
```cmd
python setup_v2.py
```

Then follow steps 1-9 above!
