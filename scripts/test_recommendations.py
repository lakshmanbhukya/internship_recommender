"""
Comprehensive Recommendation System Tests
Tests both lightweight and full modes
"""
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_lightweight_mode():
    """Test lightweight search engine"""
    print("\n" + "="*60)
    print("TEST 1: LIGHTWEIGHT MODE (512 MB)")
    print("="*60)
    
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    
    engine = get_search_engine()
    
    # Test 1: Python Developer
    print("\n[Test 1.1] Python Developer Search")
    results = engine.search(
        user_skills=["Python", "Django"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=5000,
        top_k=5
    )
    
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']}")
        print(f"   Skills: {', '.join(r['skills'][:3])}")
        print(f"   Score: {r['match_score']:.1f} | Distance: {r['distance_km']}km")
    
    # Test 2: Machine Learning
    print("\n[Test 1.2] Machine Learning Search")
    results = engine.search(
        user_skills=["Machine Learning", "Python"],
        education="B.Tech",
        city="Mumbai",
        max_distance_km=100,
        min_stipend=0,
        top_k=5
    )
    
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']}")
        print(f"   Score: {r['match_score']:.1f}")
    
    # Test 3: Marketing
    print("\n[Test 1.3] Marketing Search")
    results = engine.search(
        user_skills=["Social Media", "Content Writing"],
        education="B.Com",
        city="Delhi",
        max_distance_km=50,
        min_stipend=0,
        top_k=5
    )
    
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']}")
        print(f"   Score: {r['match_score']:.1f}")
    
    # Test 4: Remote Work
    print("\n[Test 1.4] Remote Work Search")
    results = engine.search(
        user_skills=["JavaScript", "React"],
        education="B.Tech",
        city="Remote",
        max_distance_km=1000,
        min_stipend=0,
        top_k=5
    )
    
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']} ({r['city']})")
        print(f"   Score: {r['match_score']:.1f}")
    
    # Test 5: High Stipend Filter
    print("\n[Test 1.5] High Stipend Filter (>10000)")
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=10000,
        top_k=5
    )
    
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']}")
        print(f"   Stipend: Rs.{r['stipend_min']}-{r['stipend_max']}")
        print(f"   Score: {r['match_score']:.1f}")
    
    # Don't close - keep connection for other tests
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TEST 2: EDGE CASES")
    print("="*60)
    
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    
    engine = get_search_engine()
    
    # Test 1: Single skill
    print("\n[Test 2.1] Single Skill")
    results = engine.search(
        user_skills=["Java"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=0,
        top_k=3
    )
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        has_java = any('java' in s.lower() for s in r['skills'])
        print(f"{i}. {r['role']} - Has Java: {has_java}")
    
    # Test 2: Many skills
    print("\n[Test 2.2] Many Skills (10+)")
    results = engine.search(
        user_skills=["Python", "Django", "Flask", "REST API", "PostgreSQL", 
                    "Docker", "Git", "Linux", "AWS", "Redis"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=0,
        top_k=3
    )
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} @ {r['company']}")
        print(f"   Score: {r['match_score']:.1f}")
    
    # Test 3: Unknown city
    print("\n[Test 2.3] Unknown City")
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="UnknownCity",
        max_distance_km=50,
        min_stipend=0,
        top_k=3
    )
    print(f"Results: {len(results)}")
    
    # Test 4: Very high stipend
    print("\n[Test 2.4] Very High Stipend (>50000)")
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=50000,
        top_k=3
    )
    print(f"Results: {len(results)}")
    
    # Test 5: Zero distance
    print("\n[Test 2.5] Zero Distance (Same City Only)")
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=0,
        min_stipend=0,
        top_k=3
    )
    print(f"Results: {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['role']} - City: {r['city']}")
    
    return True

def test_accuracy():
    """Test recommendation accuracy"""
    print("\n" + "="*60)
    print("TEST 3: ACCURACY TESTS")
    print("="*60)
    
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    
    engine = get_search_engine()
    
    test_cases = [
        {
            "name": "Backend Developer",
            "skills": ["Python", "Django", "REST API"],
            "expected_keywords": ["backend", "python", "django", "api", "developer", "web"]
        },
        {
            "name": "Frontend Developer",
            "skills": ["React", "JavaScript", "HTML", "CSS"],
            "expected_keywords": ["frontend", "react", "javascript", "web", "ui", "development"]
        },
        {
            "name": "Data Science",
            "skills": ["Python", "Machine Learning", "Pandas"],
            "expected_keywords": ["data", "python", "ml", "analytics", "science", "machine"]
        },
        {
            "name": "Marketing",
            "skills": ["Social Media", "Content Writing"],
            "expected_keywords": ["marketing", "social", "content", "media", "digital"]
        }
    ]
    
    total_accuracy = 0
    
    for test in test_cases:
        print(f"\n[Test 3.{test_cases.index(test)+1}] {test['name']}")
        results = engine.search(
            user_skills=test['skills'],
            education="B.Tech",
            city="Bangalore",
            max_distance_km=100,
            min_stipend=0,
            top_k=5
        )
        
        print(f"Query: {', '.join(test['skills'])}")
        print(f"Results: {len(results)}")
        
        # Check relevance
        relevant_count = 0
        for r in results:
            role_lower = r['role'].lower()
            skills_lower = ' '.join(r['skills']).lower()
            text = f"{role_lower} {skills_lower}"
            
            matches = sum(1 for kw in test['expected_keywords'] if kw in text)
            if matches >= 2:
                relevant_count += 1
        
        accuracy = (relevant_count / len(results) * 100) if results else 0
        total_accuracy += accuracy
        print(f"Relevance: {relevant_count}/{len(results)} ({accuracy:.1f}%)")
        
        # Show top 3 with match details
        for i, r in enumerate(results[:3], 1):
            role_lower = r['role'].lower()
            skills_lower = ' '.join(r['skills']).lower()
            text = f"{role_lower} {skills_lower}"
            matches = [kw for kw in test['expected_keywords'] if kw in text]
            print(f"  {i}. {r['role']} @ {r['company']} (Score: {r['match_score']:.1f})")
            print(f"     Matched: {', '.join(matches) if matches else 'None'}")
    
    avg_accuracy = total_accuracy / len(test_cases)
    print(f"\n[OVERALL ACCURACY] {avg_accuracy:.1f}%")
    
    return True

def test_performance():
    """Test performance metrics"""
    print("\n" + "="*60)
    print("TEST 4: PERFORMANCE")
    print("="*60)
    
    import time
    
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    
    engine = get_search_engine()
    
    # Warmup
    engine.search(["Python"], "B.Tech", "Bangalore", 50, 0, 5)
    
    # Test latency
    print("\n[Test 4.1] Latency Test (10 queries)")
    latencies = []
    
    for i in range(10):
        start = time.time()
        results = engine.search(
            user_skills=["Python", "Django"],
            education="B.Tech",
            city="Bangalore",
            max_distance_km=50,
            min_stipend=0,
            top_k=10
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f"  Query {i+1}: {latency:.1f}ms ({len(results)} results)")
    
    print(f"\nAverage: {sum(latencies)/len(latencies):.1f}ms")
    print(f"Min: {min(latencies):.1f}ms")
    print(f"Max: {max(latencies):.1f}ms")
    
    # Test different result sizes
    print("\n[Test 4.2] Result Size Impact")
    for top_k in [5, 10, 20, 50]:
        start = time.time()
        results = engine.search(
            user_skills=["Python"],
            education="B.Tech",
            city="Bangalore",
            max_distance_km=100,
            min_stipend=0,
            top_k=top_k
        )
        latency = (time.time() - start) * 1000
        print(f"  top_k={top_k}: {latency:.1f}ms ({len(results)} results)")
    
    return True

def test_health():
    """Test health check"""
    print("\n" + "="*60)
    print("TEST 5: HEALTH CHECK")
    print("="*60)
    
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    
    engine = get_search_engine()
    health = engine.get_health()
    
    print("\nHealth Status:")
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # Close only at the very end
    engine.close()
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("INTERNSHIP RECOMMENDATION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Lightweight Mode", test_lightweight_mode),
        ("Edge Cases", test_edge_cases),
        ("Accuracy", test_accuracy),
        ("Performance", test_performance),
        ("Health Check", test_health)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n[FAIL] {name} FAILED: {e}")
            results.append((name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results:
        symbol = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\nTotal: {passed}/{len(results)} tests passed")

if __name__ == "__main__":
    main()
