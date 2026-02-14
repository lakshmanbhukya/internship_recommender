"""Quick API test"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing Internship Recommender API\n")

# Test 1: Health check
print("1. Health Check...")
try:
    r = requests.get(f"{BASE_URL}/")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=2)}\n")
except Exception as e:
    print(f"   Error: {e}\n")

# Test 2: Python + ML recommendation
print("2. Python + ML Recommendation...")
try:
    r = requests.post(f"{BASE_URL}/recommend", json={
        "skills": ["Python", "Machine Learning"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 50,
        "min_stipend": 5000
    })
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Total Results: {data['total_results']}")
    if data['recommendations']:
        top = data['recommendations'][0]
        print(f"   Top Match: {top['role']} @ {top['company']}")
        print(f"   Score: {top['match_score']:.1f}")
        print(f"   Skills: {', '.join(top['skills'][:3])}\n")
except Exception as e:
    print(f"   Error: {e}\n")

# Test 3: Java exact match
print("3. Java Exact Match Test...")
try:
    r = requests.post(f"{BASE_URL}/recommend", json={
        "skills": ["Java"],
        "education": "B.Tech",
        "city": "Remote",
        "max_distance_km": 100
    })
    data = r.json()
    print(f"   Results: {data['total_results']}")
    if data['recommendations']:
        for i, rec in enumerate(data['recommendations'][:3], 1):
            has_java = 'Java' in rec['skills'] or 'java' in str(rec['skills']).lower()
            print(f"   {i}. {rec['role']} - Java: {has_java}")
except Exception as e:
    print(f"   Error: {e}\n")

print("\nTest complete!")
