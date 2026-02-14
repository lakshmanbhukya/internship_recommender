"""Verify critical fixes without loading models"""
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH

print("=" * 60)
print("VERIFYING CRITICAL FIXES")
print("=" * 60)

# 1. Check FTS5 table
print("\n1. Checking FTS5 table...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_internships'")
if cursor.fetchone():
    count = conn.execute("SELECT COUNT(*) FROM fts_internships").fetchone()[0]
    print(f"   [OK] FTS5 table exists with {count} records")
else:
    print("   [MISSING] FTS5 table missing")
    print("   -> Run: python database/create_database.py")

# 2. Check API file uses hybrid search
print("\n2. Checking API uses hybrid search...")
with open("api/main.py", "r") as f:
    content = f.read()
    if "from api.hybrid_search import get_engine" in content:
        print("   [OK] API imports hybrid search engine")
    else:
        print("   [MISSING] API still using old engine")
    
    if "logger" in content:
        print("   [OK] Logging configured")
    else:
        print("   [MISSING] No logging")
    
    if "shutdown_event" in content:
        print("   [OK] Shutdown handler added")
    else:
        print("   [MISSING] No shutdown handler")

# 3. Check schemas have validation
print("\n3. Checking input validation...")
with open("api/schemas.py", "r") as f:
    content = f.read()
    if "@validator" in content:
        print("   [OK] Input validation added")
    else:
        print("   [MISSING] No validation")

# 4. Check hybrid_search has FTS5
print("\n4. Checking FTS5 search method...")
with open("api/hybrid_search.py", "r", encoding="utf-8") as f:
    content = f.read()
    if "_fts5_search" in content:
        print("   [OK] FTS5 search method exists")
    else:
        print("   [MISSING] FTS5 method missing")
    
    if "bm25" in content:
        print("   [OK] BM25 ranking implemented")
    else:
        print("   [MISSING] No BM25")

conn.close()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nTo test API:")
print("  python api/main.py")
print("\nTo test search:")
print("  curl -X POST http://localhost:8000/recommend \\")
print('    -H "Content-Type: application/json" \\')
print('    -d \'{"skills":["Python"],"education":"B.Tech","city":"Bangalore"}\'')
