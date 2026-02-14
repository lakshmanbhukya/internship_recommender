import sqlite3
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

def create_database():
    print("🚀 Creating SQLite database...")
    
    # Check files exist
    metadata_path = DATA_DIR / "internship_metadata.csv"
    embeddings_path = DATA_DIR / "internship_embeddings.npy"
    
    if not metadata_path.exists() or not embeddings_path.exists():
        print("❌ Missing files. Run Colab notebook first and place files in data/")
        return None
    
    df = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)
    print(f"📊 Loaded {len(df)} internships, embeddings: {embeddings.shape}")
    
    # Load distance matrix
    if CITY_DISTANCE_MATRIX.exists():
        with open(CITY_DISTANCE_MATRIX, 'r') as f:
            city_distances = json.load(f)
        print(f"📍 Loaded distance matrix: {len(city_distances)} cities")
    else:
        city_distances = {}
        print("⚠️ No distance matrix")
    
    # Create database
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    # Try to load sqlite-vec
    try:
        conn.enable_load_extension(True)
        conn.load_extension('vec0')
        print("✅ sqlite-vec loaded")
        has_vec = True
    except:
        print("⚠️ sqlite-vec not available, using fallback")
        has_vec = False
    
    # Create tables
    print("📋 Creating tables...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS internships (
            id TEXT PRIMARY KEY,
            profile TEXT NOT NULL,
            company TEXT,
            location_original TEXT,
            location_normalized TEXT,
            stipend_min INTEGER,
            stipend_max INTEGER,
            duration_months INTEGER,
            education_req TEXT,
            skills TEXT,
            perks TEXT,
            apply_by DATE,
            freshness_score REAL,
            embedding BLOB
        )
    """)
    
    # Create indexes
    print("⚡ Creating indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_location ON internships(location_normalized)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_education ON internships(education_req)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stipend ON internships(stipend_min)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_freshness ON internships(freshness_score DESC)")
    
    # Create FTS5 virtual table
    print("📝 Creating FTS5 full-text search index...")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_internships 
        USING fts5(id UNINDEXED, profile, skills)
    """)
    
    # Insert data
    print("📥 Inserting data...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        # Insert into main table
        conn.execute("""
            INSERT OR REPLACE INTO internships VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row['internship_id'],
            row['profile'],
            row['company'],
            row['Location'],
            row['location_normalized'],
            int(row['stipend_min']),
            int(row['stipend_max']),
            int(row['duration_months']),
            row['education_normalized'],
            json.dumps(eval(row['skills_clean']) if isinstance(row['skills_clean'], str) else row['skills_clean']),
            row['Perks'],
            row['Apply by Date'],
            float(row['freshness_score']),
            embeddings[idx].tobytes()
        ))
        
        # Insert into FTS5 table
        skills_list = eval(row['skills_clean']) if isinstance(row['skills_clean'], str) else row['skills_clean']
        skills_text = ' '.join(skills_list) if isinstance(skills_list, list) else str(skills_list)
        conn.execute("""
            INSERT INTO fts_internships(id, profile, skills)
            VALUES (?, ?, ?)
        """, (row['internship_id'], row['profile'], skills_text))
    
    conn.commit()
    conn.close()
    
    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅✅✅ Database created!")
    print(f"💾 Path: {DB_PATH}")
    print(f"📊 Records: {len(df)}")
    print(f"📦 Size: {db_size:.2f} MB")
    
    return DB_PATH

def test_database():
    print("\n🧪 Testing database...")
    conn = sqlite3.connect(DB_PATH)
    
    count = conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
    print(f"✓ Total: {count}")
    
    sample = conn.execute("""
        SELECT id, profile, company, location_normalized, stipend_min
        FROM internships LIMIT 3
    """).fetchall()
    print(f"✓ Sample:")
    for row in sample:
        print(f"  - {row[1]} at {row[2]} ({row[3]}) - ₹{row[4]}")
    
    conn.close()
    print("✅ Tests passed!")

if __name__ == "__main__":
    create_database()
    test_database()
