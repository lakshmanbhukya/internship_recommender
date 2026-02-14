"""Test industry-grade fixes"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def test_api_imports():
    """Test API uses hybrid search"""
    print("Testing API imports...")
    from api.main import engine
    from api.hybrid_search import HybridSearchEngine
    print("✓ API imports correct")

def test_fts5():
    """Test FTS5 search"""
    print("\nTesting FTS5 search...")
    import sqlite3
    from config.settings import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_internships'")
        if cursor.fetchone():
            print("✓ FTS5 table exists")
            
            # Test search
            cursor = conn.execute("SELECT COUNT(*) FROM fts_internships")
            count = cursor.fetchone()[0]
            print(f"✓ FTS5 has {count} records")
        else:
            print("✗ FTS5 table missing - run: python database/create_database.py")
    except Exception as e:
        print(f"✗ FTS5 error: {e}")
    finally:
        conn.close()

def test_validation():
    """Test input validation"""
    print("\nTesting input validation...")
    from api.schemas import UserProfile
    from pydantic import ValidationError
    
    try:
        # Should fail - empty skill
        UserProfile(skills=[""], education="B.Tech", city="Bangalore")
        print("✗ Validation failed - empty skills allowed")
    except ValidationError:
        print("✓ Empty skills rejected")
    
    try:
        # Should pass
        profile = UserProfile(skills=["Python", "ML"], education="B.Tech", city="Bangalore")
        print(f"✓ Valid profile accepted: {profile.skills}")
    except Exception as e:
        print(f"✗ Valid profile rejected: {e}")

def test_hybrid_search():
    """Test hybrid search engine"""
    print("\nTesting hybrid search...")
    try:
        from api.hybrid_search import get_engine
        engine = get_engine()
        
        results = engine.search(
            user_skills=["Python", "Machine Learning"],
            education="B.Tech",
            city="Bangalore",
            max_distance_km=50,
            min_stipend=0,
            top_k=5
        )
        
        print(f"✓ Search returned {len(results)} results")
        if results:
            print(f"  Top match: {results[0]['role']} @ {results[0]['company']} (score: {results[0]['match_score']:.1f})")
        
        engine.close()
    except Exception as e:
        print(f"✗ Search failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("INDUSTRY-GRADE FIXES VERIFICATION")
    print("=" * 60)
    
    test_api_imports()
    test_fts5()
    test_validation()
    test_hybrid_search()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
