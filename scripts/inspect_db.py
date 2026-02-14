import sqlite3
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH

def inspect_database():
    print("Inspecting database structure...\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check tables
    print("Tables in database:")
    for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        print(f"  - {row[0]}")
    
    # Check internships table structure
    print("\nInternships table schema:")
    for row in cursor.execute("PRAGMA table_info(internships)"):
        col_id, name, col_type, not_null, default, pk = row
        print(f"  {col_id}: {name} ({col_type})")
    
    # Sample embedding check
    print("\nSample embedding check:")
    sample = cursor.execute("SELECT id, embedding FROM internships LIMIT 1").fetchone()
    if sample:
        emb_id, emb_bytes = sample
        print(f"  ID: {emb_id}")
        print(f"  Embedding size: {len(emb_bytes)} bytes")
        # Try to decode as numpy array
        try:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            print(f"  Decoded shape: {emb.shape}")
            print(f"  First 5 values: {emb[:5]}")
            print(f"  [OK] Embeddings are valid!")
        except Exception as e:
            print(f"  [WARNING] Decode failed: {e}")
    else:
        print("  [WARNING] No data found in internships table")
    
    # Count total records
    total = cursor.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
    print(f"\nTotal internships: {total}")
    
    conn.close()
    print("\n[OK] Inspection complete!")

if __name__ == "__main__":
    inspect_database()
