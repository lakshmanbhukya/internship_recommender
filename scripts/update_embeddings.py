import sqlite3
import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path

# Paths
DB_PATH = "database/internships.db"
NEW_EMBEDDINGS = "data/embeddings_v2.npy"
NEW_METADATA = "data/metadata_v2.csv"
ENHANCED_CSV = "data/processed/internships_enhanced.csv"
FAISS_INDEX_PATH = "data/faiss_index.bin"
ID_MAPPING_PATH = "data/id_mapping.json"

print("Step 3: Updating database with new embeddings...")

# Load new embeddings and metadata
print("Loading new embeddings...")
embeddings = np.load(NEW_EMBEDDINGS)
print(f"Loaded embeddings shape: {embeddings.shape}")

print("Loading enhanced dataset...")
df_enhanced = pd.read_csv(ENHANCED_CSV)
print(f"Loaded {len(df_enhanced)} records")

# Update database
print("\nUpdating database with new embeddings and metadata...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Add new columns if they don't exist
try:
    cursor.execute("ALTER TABLE internships ADD COLUMN role_type TEXT")
    cursor.execute("ALTER TABLE internships ADD COLUMN seniority TEXT")
    print("Added role_type and seniority columns")
except sqlite3.OperationalError:
    print("Columns already exist, skipping...")

# Update embeddings and metadata
updated = 0
for idx, row in df_enhanced.iterrows():
    if idx >= len(embeddings):
        break
    
    embedding_bytes = embeddings[idx].astype(np.float32).tobytes()
    
    cursor.execute("""
        UPDATE internships 
        SET embedding = ?, role_type = ?, seniority = ?
        WHERE id = ?
    """, (embedding_bytes, row['role_type'], row['seniority'], row['internship_id']))
    
    updated += 1
    if updated % 1000 == 0:
        print(f"  Updated {updated}/{len(df_enhanced)} records...")

conn.commit()
print(f"[SUCCESS] Updated {updated} records in database")

# Rebuild FAISS index
print("\nRebuilding FAISS index...")
cursor.execute("SELECT id, embedding FROM internships ORDER BY rowid")
rows = cursor.fetchall()

embedding_dim = 1024
faiss_embeddings = np.zeros((len(rows), embedding_dim), dtype='float32')
internship_ids = []

for i, (internship_id, emb_bytes) in enumerate(rows):
    emb = np.frombuffer(emb_bytes, dtype=np.float32)
    faiss_embeddings[i] = emb
    internship_ids.append(internship_id)

print(f"Building FAISS HNSW index with {len(faiss_embeddings)} vectors...")
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
index.add(faiss_embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
print(f"[SUCCESS] FAISS index saved to {FAISS_INDEX_PATH}")

# Save ID mapping
with open(ID_MAPPING_PATH, 'w') as f:
    json.dump({"ids": internship_ids}, f)
print(f"[SUCCESS] ID mapping saved to {ID_MAPPING_PATH}")

conn.close()

print("\n" + "="*60)
print("[COMPLETE] Step 3 finished successfully!")
print("="*60)
print("\nNext: Run test to verify improvements")
