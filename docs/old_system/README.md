# Old System Documentation

This folder contains documentation for the previous ChromaDB-based system.

## Files

- **[README_OLD.md](README_OLD.md)** - Original README for ChromaDB system
- **[SEMANTIC_UPGRADE.md](SEMANTIC_UPGRADE.md)** - Previous semantic upgrade documentation

## System Overview (Deprecated)

The old system used:
- ChromaDB for vector storage (~200MB+)
- MongoDB for metadata (optional)
- Multiple scattered modules
- Complex setup process

## Migration Status

⚠️ **This system is deprecated**

Please use the new v2.0 system with:
- SQLite-vec (35MB)
- Clean FastAPI architecture
- Automated pipeline

## Migration Path

See [../migration/START_HERE.md](../migration/START_HERE.md) for complete migration guide.

## Why Migrate?

| Aspect | Old | New | Improvement |
|--------|-----|-----|-------------|
| Size | 200MB+ | 35MB | 85% smaller |
| Startup | 10s | 5s | 50% faster |
| Architecture | Scattered | Layered | Maintainable |
| Deployment | Complex | Simple | Easy |

## Old Files to Delete

After successful migration, delete:
```
chroma/
vector_store/
embeddings/
semantic_recommender/
resume_parser/
utils/
connection.py
recommender.py
main.py (old version)
migrate_data.py
setup_semantic.py
test_semantic.py
tfidf_vectorizer.joblib
```

---

**For new system, see main [README.md](../../README.md)**
