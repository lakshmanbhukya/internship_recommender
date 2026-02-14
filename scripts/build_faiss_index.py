"""
Build FAISS index from existing SQLite DB - NO retraining needed!
Run this ONCE after DB creation.
"""
import sqlite3
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, DATA_DIR

# Check if faiss is installed
try:
    import faiss
except ImportError:
    print("❌ FAISS not installed!")
    print("Install with: pip install faiss-cpu")
    sys.exit(1)

FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_NPY_PATH = DATA_DIR / "embeddings_backup.npy"
ID_MAPPING_PATH = DATA_DIR / "id_mapping.json"

def build_faiss_from_db():
    print("Building FAISS index from existing SQLite DB...")
    
    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch ALL embeddings
    print("Loading embeddings from database...")
    cursor.execute("SELECT id, embedding FROM internships ORDER BY rowid")
    rows = cursor.fetchall()
    
    if not rows:
        raise ValueError("No embeddings found in internships table!")
    
    print(f"[OK] Loaded {len(rows)} embeddings")
    
    # Convert BLOBs to numpy array
    embedding_dim = 1024  # BGE-M3 dimension
    embeddings = np.zeros((len(rows), embedding_dim), dtype='float32')
    internship_ids = []
    
    print("Converting embeddings to numpy array...")
    for i, (internship_id, emb_bytes) in enumerate(rows):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)} ({(i + 1)/len(rows)*100:.1f}%)")
        
        # Convert bytes → float32 array
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        
        if emb.shape[0] != embedding_dim:
            raise ValueError(f"Embedding {i} has wrong dimension: {emb.shape[0]} (expected {embedding_dim})")
        
        embeddings[i] = emb
        internship_ids.append(internship_id)
    
    print(f"[OK] Converted to numpy array: {embeddings.shape}")
    
    # Build FAISS HNSW index
    print("Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 = HNSW M parameter
    index.hnsw.efConstruction = 200  # Higher = better quality during build
    index.hnsw.efSearch = 64         # Higher = better quality during search
    
    print("  Adding vectors to index...")
    index.add(embeddings)
    
    print(f"[OK] FAISS index built: {index.ntotal} vectors")
    
    # Save index to disk
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    
    # Optional: Save embeddings.npy backup
    np.save(str(EMBEDDINGS_NPY_PATH), embeddings)
    print(f"Embeddings backup saved to: {EMBEDDINGS_NPY_PATH}")
    
    # Save ID mapping
    import json
    with open(ID_MAPPING_PATH, 'w') as f:
        json.dump({"ids": internship_ids}, f)
    print(f"ID mapping saved to: {ID_MAPPING_PATH}")
    
    conn.close()
    
    # Verify load
    print("\nVerifying index load...")
    index2 = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"[OK] Verified: {index2.ntotal} vectors loaded")
    
    # Get file sizes
    faiss_size = FAISS_INDEX_PATH.stat().st_size / (1024 * 1024)
    emb_size = EMBEDDINGS_NPY_PATH.stat().st_size / (1024 * 1024)
    
    print(f"\nFile sizes:")
    print(f"  - FAISS index: {faiss_size:.2f} MB")
    print(f"  - Embeddings backup: {emb_size:.2f} MB")
    
    print("\n[SUCCESS] FAISS index built successfully! Ready for inference.")
    return FAISS_INDEX_PATH, internship_ids

if __name__ == "__main__":
    build_faiss_from_db()
