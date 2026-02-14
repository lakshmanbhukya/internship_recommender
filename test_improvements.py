import os
os.environ["LIGHTWEIGHT_MODE"] = "true"

from api.lightweight_search import LightweightSearchEngine

engine = LightweightSearchEngine()

print("\n=== TEST: Python + Django ===")
results = engine.search(["Python", "Django"], "B.Tech", "Bangalore", 50, 5000, 5)
print(f"Results: {len(results)}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['role']} @ {r['company']}")
    print(f"   Skills: {', '.join(r['skills'][:3])}")
    print(f"   Score: {r['match_score']}")

print("\n=== TEST: Machine Learning ===")
results = engine.search(["Machine Learning"], "B.Tech", "Mumbai", 100, 0, 5)
print(f"Results: {len(results)}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['role']} - Score: {r['match_score']}")

print("\n=== TEST: JavaScript ===")
results = engine.search(["JavaScript"], "B.Tech", "Bangalore", 50, 0, 5)
print(f"Results: {len(results)}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['role']} - Score: {r['match_score']}")

engine.close()
print("\n[OK] Tests complete!")
