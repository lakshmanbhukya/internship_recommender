# Migration Documentation

Complete guide for migrating from ChromaDB to SQLite-vec system.

## Quick Navigation

### Getting Started
- **[START_HERE.md](START_HERE.md)** - Step-by-step migration guide (START HERE!)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet

### Understanding the System
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams
- **[STRUCTURE_COMPARISON.md](STRUCTURE_COMPARISON.md)** - Old vs new comparison

### Detailed Guides
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Complete migration instructions
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Deliverables and checklist

### Reference
- **[INDEX.md](INDEX.md)** - Master documentation index
- **[README_V2.md](README_V2.md)** - Detailed v2.0 README
- **[ORIGINAL_SPEC.md](ORIGINAL_SPEC.md)** - Original specification document

## Migration Steps (30 minutes)

1. **Setup** (5 min)
   - Get Kaggle credentials
   - Install dependencies: `pip install -r requirements-new.txt`

2. **Data Pipeline** (10 min)
   ```bash
   python scripts/download_dataset.py
   python scripts/preprocess_data.py
   python models/geocode_cities.py
   ```

3. **Embeddings** (5 min on Colab)
   - Upload `notebooks/02_embedding_generation.ipynb` to Colab
   - Set GPU runtime (T4)
   - Download files to `data/`

4. **Database** (2 min)
   ```bash
   python database/create_database.py
   ```

5. **Test** (2 min)
   ```bash
   python api/main.py
   python test_api_v2.py
   ```

## Key Improvements

- **85% smaller**: 35MB vs 200MB+
- **50% faster**: 5s startup vs 10s
- **Cleaner**: Layered architecture
- **Simpler**: Single database file

## Files to Delete After Migration

```
chroma/
vector_store/
embeddings/
semantic_recommender/
resume_parser/
utils/
connection.py
recommender.py
main.py (old)
migrate_data.py
setup_semantic.py
test_semantic.py
tfidf_vectorizer.joblib
requirements.txt (replace with requirements-new.txt)
```

## Support

For issues, check:
1. [START_HERE.md](START_HERE.md) - Troubleshooting section
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
3. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Detailed guide

## Success Criteria

✅ API responds at http://localhost:8000/
✅ Health check shows database_connected: true
✅ Recommendations return in <100ms
✅ Match scores between 50-100

---

**Ready? Start with [START_HERE.md](START_HERE.md)**
