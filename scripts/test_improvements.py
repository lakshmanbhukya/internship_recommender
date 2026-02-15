import sys
sys.path.append('.')
from api.hybrid_search import get_engine

print("="*70)
print("TESTING IMPROVED RECOMMENDATIONS")
print("="*70)

engine = get_engine()

# Test Case 1: Backend Developer (Python/Django)
print("\n" + "="*70)
print("Test 1: Backend Developer (Python/Django)")
print("="*70)
results = engine.search(
    user_skills=["Python", "Django", "REST API", "PostgreSQL"],
    education="B.Tech",
    city="Bangalore",
    max_distance_km=50,
    min_stipend=10000,
    top_k=5
)

print(f"\nFound {len(results)} recommendations:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['role']} @ {r['company']}")
    print(f"   Location: {r['city']} ({r['distance_km']}km away)")
    print(f"   Stipend: Rs.{r['stipend_min']:,}-{r['stipend_max']:,}/month")
    print(f"   Skills: {', '.join(r['skills'][:5])}")
    print(f"   Match Score: {r['match_score']:.1f}%")

# Test Case 2: DevOps Engineer
print("\n" + "="*70)
print("Test 2: DevOps Engineer (AWS/Docker/K8s)")
print("="*70)
results = engine.search(
    user_skills=["AWS", "Docker", "Kubernetes", "Python", "Linux"],
    education="B.Tech",
    city="Bangalore",
    max_distance_km=50,
    min_stipend=12000,
    top_k=5
)

print(f"\nFound {len(results)} recommendations:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['role']} @ {r['company']}")
    print(f"   Location: {r['city']} ({r['distance_km']}km away)")
    print(f"   Stipend: Rs.{r['stipend_min']:,}-{r['stipend_max']:,}/month")
    print(f"   Skills: {', '.join(r['skills'][:5])}")
    print(f"   Match Score: {r['match_score']:.1f}%")

# Test Case 3: Full Stack (MERN)
print("\n" + "="*70)
print("Test 3: Full Stack Developer (MERN)")
print("="*70)
results = engine.search(
    user_skills=["MongoDB", "Express", "React", "Node.js"],
    education="B.Tech",
    city="Mumbai",
    max_distance_km=50,
    min_stipend=10000,
    top_k=5
)

print(f"\nFound {len(results)} recommendations:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['role']} @ {r['company']}")
    print(f"   Location: {r['city']} ({r['distance_km']}km away)")
    print(f"   Stipend: Rs.{r['stipend_min']:,}-{r['stipend_max']:,}/month")
    print(f"   Skills: {', '.join(r['skills'][:5])}")
    print(f"   Match Score: {r['match_score']:.1f}%")

# Test Case 4: UI/UX Designer
print("\n" + "="*70)
print("Test 4: UI/UX Designer (Figma/Sketch)")
print("="*70)
results = engine.search(
    user_skills=["Figma", "UI Design", "UX Design", "Wireframing"],
    education="Any",
    city="Bangalore",
    max_distance_km=50,
    min_stipend=8000,
    top_k=5
)

print(f"\nFound {len(results)} recommendations:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['role']} @ {r['company']}")
    print(f"   Location: {r['city']} ({r['distance_km']}km away)")
    print(f"   Stipend: Rs.{r['stipend_min']:,}-{r['stipend_max']:,}/month")
    print(f"   Skills: {', '.join(r['skills'][:5])}")
    print(f"   Match Score: {r['match_score']:.1f}%")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nExpected improvements:")
print("- Backend roles should show Django/Flask, not ML/AI")
print("- DevOps should show AWS/Docker, not embedded systems")
print("- Match scores should be 75-90% (up from 51-63%)")
print("- Distance should show real km (not 0.0km)")
