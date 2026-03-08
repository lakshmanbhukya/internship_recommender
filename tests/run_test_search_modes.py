"""
Unified Test Script - Compare Lightweight vs BGE-M3
Run: python test_search_modes.py [lightweight|bge-m3|both]
"""
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

PROFILES = [
    {"name": "Backend Dev", "skills": ["Python", "Django", "REST API"], "education": "B.Tech", "city": "Bangalore"},
    {"name": "Frontend Dev", "skills": ["React", "JavaScript", "HTML"], "education": "B.Tech", "city": "Mumbai"},
    {"name": "Data Scientist", "skills": ["Python", "Machine Learning", "Pandas"], "education": "M.Tech", "city": "Bangalore"},
    {"name": "Digital Marketer", "skills": ["Social Media", "Content Writing", "SEO"], "education": "B.Com", "city": "Delhi"},
    {"name": "Full Stack Dev", "skills": ["MERN", "Node.js", "React"], "education": "B.Tech", "city": "Hyderabad"},
]

def test_mode(mode_name, lightweight=True):
    os.environ["LIGHTWEIGHT_MODE"] = "true" if lightweight else "false"
    from api.engine_selector import get_search_engine
    
    print(f"\n{'='*60}")
    print(f"Testing {mode_name}")
    print(f"{'='*60}")
    
    start_load = time.time()
    engine = get_search_engine()
    load_time = time.time() - start_load
    print(f"Load Time: {load_time:.2f}s")
    
    total_time = 0
    for profile in PROFILES:
        start = time.time()
        results = engine.search(
            user_skills=profile["skills"],
            education=profile["education"],
            city=profile["city"],
            max_distance_km=50,
            min_stipend=5000,
            top_k=5
        )
        query_time = time.time() - start
        total_time += query_time
        
        print(f"\n{profile['name']}: {len(results)} results ({query_time*1000:.0f}ms)")
        if results:
            print(f"  Top: {results[0]['role']} (Score: {results[0]['match_score']:.1f})")
    
    avg_time = total_time / len(PROFILES)
    print(f"\nAvg Query Time: {avg_time*1000:.0f}ms")
    print(f"Total Time: {total_time:.2f}s")
    
    engine.close()
    return load_time, avg_time

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    if mode in ["lightweight", "both"]:
        light_load, light_query = test_mode("LIGHTWEIGHT MODE", lightweight=True)
    
    if mode in ["bge-m3", "both"]:
        bge_load, bge_query = test_mode("BGE-M3 MODE", lightweight=False)
    
    if mode == "both":
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Load Time:  Lightweight {light_load:.2f}s vs BGE-M3 {bge_load:.2f}s ({bge_load/light_load:.0f}x slower)")
        print(f"Query Time: Lightweight {light_query*1000:.0f}ms vs BGE-M3 {bge_query*1000:.0f}ms ({bge_query/light_query:.0f}x slower)")

if __name__ == "__main__":
    main()
