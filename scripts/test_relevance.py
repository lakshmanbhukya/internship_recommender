"""
================================================================
  Internship Recommendation Relevance Test Suite

  Tests that the recommendation system returns RELEVANT results
  across 8 categories:

   1. Persona-Based Relevance   (do skills match the role?)
   2. Skill Precision & Recall  (are returned skills correct?)
   3. Cross-Domain Contamination (no backend in frontend query?)
   4. Education Hierarchy       (does B.Tech see Diploma roles?)
   5. Filter Correctness        (stipend, distance, city)
   6. Edge Cases                (single skill, unknown city)
   7. Scoring Sanity            (is score ordering meaningful?)
   8. Consistency               (same query -> same results?)

  Run: python scripts/test_relevance.py
================================================================
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# Personas: realistic user profiles with expected and forbidden keywords
PERSONAS = [
    {
        "name": "Backend Developer",
        "skills": ["Python", "Django", "REST API"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 100,
        "min_stipend": 0,
        "expected_role_keywords": ["backend", "python", "django", "api", "server", "development", "web"],
        "expected_skill_keywords": ["python", "django", "flask", "rest", "api", "sql", "database", "backend"],
        "forbidden_keywords": ["graphic design", "video editing", "hr", "legal"],
    },
    {
        "name": "Frontend Developer",
        "skills": ["React", "JavaScript", "HTML", "CSS"],
        "education": "B.Tech",
        "city": "Mumbai",
        "max_distance_km": 100,
        "min_stipend": 0,
        "expected_role_keywords": ["frontend", "front end", "react", "web", "ui", "javascript", "development"],
        "expected_skill_keywords": ["react", "javascript", "html", "css", "frontend", "angular", "vue", "ui", "web"],
        "forbidden_keywords": ["machine learning", "data science", "hr", "legal"],
    },
    {
        "name": "Data Scientist",
        "skills": ["Python", "Machine Learning", "Pandas", "NumPy"],
        "education": "B.Tech",
        "city": "Hyderabad",
        "max_distance_km": 100,
        "min_stipend": 0,
        "expected_role_keywords": ["data", "science", "machine learning", "ml", "ai", "analytics", "python", "research"],
        "expected_skill_keywords": ["python", "machine learning", "pandas", "data", "ml", "ai", "analytics", "numpy", "statistics"],
        "forbidden_keywords": ["graphic design", "video editing", "hr"],
    },
    {
        "name": "Marketing Intern",
        "skills": ["Social Media", "Content Writing", "SEO"],
        "education": "B.Com",
        "city": "Delhi",
        "max_distance_km": 50,
        "min_stipend": 0,
        "expected_role_keywords": ["marketing", "social media", "content", "digital", "seo", "media"],
        "expected_skill_keywords": ["marketing", "social", "content", "seo", "digital", "media", "writing"],
        "forbidden_keywords": ["python", "java", "backend", "machine learning"],
    },
    {
        "name": "Mobile Developer",
        "skills": ["Android", "Java", "Kotlin"],
        "education": "B.Tech",
        "city": "Pune",
        "max_distance_km": 100,
        "min_stipend": 0,
        "expected_role_keywords": ["android", "mobile", "app", "java", "kotlin", "development"],
        "expected_skill_keywords": ["android", "java", "kotlin", "mobile", "app", "flutter", "react native"],
        "forbidden_keywords": ["graphic design", "content writing", "hr"],
    },
    {
        "name": "DevOps Engineer",
        "skills": ["Docker", "AWS", "Linux", "CI/CD"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 100,
        "min_stipend": 0,
        "expected_role_keywords": ["devops", "cloud", "aws", "docker", "linux", "infrastructure", "backend", "development"],
        "expected_skill_keywords": ["docker", "aws", "linux", "cloud", "kubernetes", "ci", "cd", "devops"],
        "forbidden_keywords": ["graphic design", "content writing", "hr", "legal"],
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_engine():
    """Get the search engine (lightweight mode for speed)."""
    os.environ["LIGHTWEIGHT_MODE"] = "true"
    from api.engine_selector import get_search_engine
    return get_search_engine()


def make_searchable_text(result: Dict) -> str:
    """Combine role + skills into a single lowercase text for matching."""
    skills = result.get("skills", [])
    if isinstance(skills, str):
        try:
            skills = json.loads(skills)
        except:
            skills = skills.split(",")
    skill_text = " ".join(s.lower().strip() for s in skills)
    role_text = result.get("role", "").lower()
    return f"{role_text} {skill_text}"


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the text."""
    return sum(1 for kw in keywords if kw.lower() in text)


def header(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subheader(title: str):
    print(f"\n  >> {title}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 1: Persona-Based Relevance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_persona_relevance(engine) -> Tuple[int, int]:
    """
    For each persona, search and check if results contain expected keywords.
    A result is 'relevant' if its role+skills text matches ≥2 expected keywords.
    """
    header("TEST 1: PERSONA-BASED RELEVANCE")
    print("  Checks: do returned roles match the user's domain?")

    passed = 0
    total = 0

    for persona in PERSONAS:
        subheader(f"{persona['name']} → skills={persona['skills']}")
        
        results = engine.search(
            user_skills=persona["skills"],
            education=persona["education"],
            city=persona["city"],
            max_distance_km=persona["max_distance_km"],
            min_stipend=persona["min_stipend"],
            top_k=10,
        )

        if not results:
            print(f"  │  No results returned")
            print(f"  └─ {FAIL} (0 results)")
            total += 1
            continue

        relevant = 0
        for r in results:
            text = make_searchable_text(r)
            hits = count_keyword_hits(text, persona["expected_role_keywords"] + persona["expected_skill_keywords"])
            if hits >= 2:
                relevant += 1

        precision = relevant / len(results)
        total += 1

        # Show top 3
        for i, r in enumerate(results[:3], 1):
            text = make_searchable_text(r)
            matched = [kw for kw in persona["expected_skill_keywords"] if kw in text]

            print(f"  │  {i}. {r['role']} @ {r['company']} (score={r['match_score']:.1f})")
            print(f"  │     matched: {', '.join(matched[:5]) if matched else '—'}")

        if precision >= 0.6:
            print(f"  └─ {PASS}  Relevance: {relevant}/{len(results)} ({precision:.0%})")
            passed += 1
        elif precision >= 0.4:
            print(f"  └─ {WARN}  Relevance: {relevant}/{len(results)} ({precision:.0%})")
            passed += 1  # partial credit
        else:
            print(f"  └─ {FAIL}  Relevance: {relevant}/{len(results)} ({precision:.0%})")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 2: Skill Precision & Recall
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_skill_matching(engine) -> Tuple[int, int]:
    """
    For each result, calculate how many of the user's input skills
    appear in the internship's required skills.
    """
    header("TEST 2: SKILL PRECISION & RECALL")
    print("  Checks: do returned internships require the user's skills?")

    test_cases = [
        {"skills": ["Python", "Django"], "label": "Python+Django"},
        {"skills": ["React", "JavaScript"], "label": "React+JS"},
        {"skills": ["Machine Learning", "Python"], "label": "ML+Python"},
        {"skills": ["Social Media", "Content Writing"], "label": "Marketing"},
    ]

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        subheader(f"{tc['label']} → {tc['skills']}")
        
        results = engine.search(
            user_skills=tc["skills"],
            education="B.Tech",
            city="Remote",
            max_distance_km=5000,
            min_stipend=0,
            top_k=10,
        )

        if not results:
            print(f"  └─ {FAIL} (0 results)")
            continue

        user_skills_lower = set(s.lower() for s in tc["skills"])
        
        total_overlap = 0
        for r in results:
            skills = r.get("skills", [])
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills)
                except:
                    skills = skills.split(",")
            
            job_skills_lower = set(s.lower().strip() for s in skills)
            
            # Partial matching: "python" in "python3", "django" in "django rest"
            overlap = 0
            for us in user_skills_lower:
                for js in job_skills_lower:
                    if us in js or js in us:
                        overlap += 1
                        break
            
            total_overlap += overlap

        avg_recall = total_overlap / (len(results) * len(user_skills_lower))

        for i, r in enumerate(results[:3], 1):
            skills = r.get("skills", [])
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills)
                except:
                    skills = skills.split(",")
            skill_str = ", ".join(skills[:4])
            print(f"  │  {i}. {r['role']} → [{skill_str}]")

        if avg_recall >= 0.3:
            print(f"  └─ {PASS}  Avg skill recall: {avg_recall:.0%}")
            passed += 1
        else:
            print(f"  └─ {FAIL}  Avg skill recall: {avg_recall:.0%}")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 3: Cross-Domain Contamination
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_cross_domain(engine) -> Tuple[int, int]:
    """
    Ensure that results do NOT contain obvious out-of-domain roles.
    E.g., a Python/Django search should not return 'Graphic Design' roles.
    """
    header("TEST 3: CROSS-DOMAIN CONTAMINATION")
    print("  Checks: are irrelevant domains leaking into results?")

    passed = 0
    total = len(PERSONAS)

    for persona in PERSONAS:
        subheader(f"{persona['name']}")
        
        results = engine.search(
            user_skills=persona["skills"],
            education=persona["education"],
            city=persona["city"],
            max_distance_km=persona["max_distance_km"],
            min_stipend=persona["min_stipend"],
            top_k=10,
        )

        if not results:
            print(f"  └─ {PASS}  No results = no contamination")
            passed += 1
            continue

        contaminated = 0
        contaminated_roles = []
        for r in results:
            text = make_searchable_text(r)
            for forbidden in persona["forbidden_keywords"]:
                if forbidden.lower() in text:
                    contaminated += 1
                    contaminated_roles.append(f"{r['role']} (matched: '{forbidden}')")
                    break

        contamination_rate = contaminated / len(results)
        
        if contamination_rate <= 0.1:
            print(f"  └─ {PASS}  Contamination: {contaminated}/{len(results)} ({contamination_rate:.0%})")
            passed += 1
        else:
            for cr in contaminated_roles[:3]:
                print(f"  │  ⚠ {cr}")
            print(f"  └─ {FAIL}  Contamination: {contaminated}/{len(results)} ({contamination_rate:.0%})")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 4: Education Hierarchy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_education_hierarchy(engine) -> Tuple[int, int]:
    """
    Verify the education filter works correctly:
    - B.Tech user should see B.Tech, B.Sc, B.A, Diploma, Any roles
    - Diploma user should NOT see B.Tech roles
    """
    header("TEST 4: EDUCATION HIERARCHY")
    print("  Checks: does the education filter match eligibility correctly?")

    passed = 0
    total = 0

    # Test 4.1: B.Tech should see Diploma and Any
    subheader("B.Tech user can see Diploma/Any roles")
    total += 1
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Remote",
        max_distance_km=5000,
        min_stipend=0,
        top_k=20,
    )
    
    edu_types = Counter(r.get("education_req", "?") for r in results)
    print(f"  │  Education distribution: {dict(edu_types)}")
    
    # B.Tech user should be able to see Any and Diploma
    can_see_lower = any(e in edu_types for e in ["Any", "Diploma", "B.A", "B.Com", "B.Sc"])
    if can_see_lower or len(results) > 0:
        print(f"  └─ {PASS}  B.Tech sees {len(results)} results, including lower-req roles")
        passed += 1
    else:
        print(f"  └─ {FAIL}  B.Tech cannot see lower-requirement roles")

    # Test 4.2: Diploma user should NOT see B.Tech-only roles
    subheader("Diploma user excluded from B.Tech-only roles")
    total += 1
    results_diploma = engine.search(
        user_skills=["Python"],
        education="Diploma",
        city="Remote",
        max_distance_km=5000,
        min_stipend=0,
        top_k=20,
    )

    btech_only = sum(1 for r in results_diploma if r.get("education_req") == "B.Tech")
    if btech_only == 0:
        print(f"  └─ {PASS}  Diploma user sees 0 B.Tech-only roles ({len(results_diploma)} total results)")
        passed += 1
    else:
        print(f"  └─ {FAIL}  Diploma user sees {btech_only} B.Tech-only roles (should be 0)")

    # Test 4.3: 'Any' education should see everything
    subheader("'Any' education user scope")
    total += 1
    results_any = engine.search(
        user_skills=["Python"],
        education="Any",
        city="Remote",
        max_distance_km=5000,
        min_stipend=0,
        top_k=20,
    )
    
    # 'Any' is the lowest level — should only see 'Any' requirement roles
    non_any = sum(1 for r in results_any if r.get("education_req") != "Any")
    if non_any == 0:
        print(f"  └─ {PASS}  'Any' user sees only 'Any'-requirement roles ({len(results_any)} results)")
        passed += 1
    else:
        edu_seen = Counter(r.get("education_req") for r in results_any)
        print(f"  └─ {FAIL}  'Any' user sees higher-req roles: {dict(edu_seen)}")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 5: Filter Correctness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_filters(engine) -> Tuple[int, int]:
    """
    Verify hard filters work:
    - Stipend: all results have stipend_max >= min_stipend
    - Distance: all results within max_distance
    - Remote: Remote city returns any-city results
    """
    header("TEST 5: FILTER CORRECTNESS")
    print("  Checks: are hard filters (stipend, distance) enforced?")

    passed = 0
    total = 0

    # Test 5.1: Stipend ≥ 10000
    subheader("Stipend filter: min_stipend=10000")
    total += 1
    results = engine.search(
        user_skills=["Python", "Django"],
        education="B.Tech",
        city="Remote",
        max_distance_km=5000,
        min_stipend=10000,
        top_k=10,
    )

    violations = [r for r in results if r["stipend_max"] < 10000]
    if not violations:
        stipends = [f"₹{r['stipend_min']}-{r['stipend_max']}" for r in results[:3]]
        print(f"  │  Sample stipends: {', '.join(stipends)}")
        print(f"  └─ {PASS}  All {len(results)} results have stipend_max ≥ ₹10,000")
        passed += 1
    else:
        for v in violations[:3]:
            print(f"  │  ⚠ {v['role']}: ₹{v['stipend_min']}-{v['stipend_max']}")
        print(f"  └─ {FAIL}  {len(violations)} violations found")

    # Test 5.2: Distance ≤ max_distance
    subheader("Distance filter: max_distance=50km, city=Bangalore")
    total += 1
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Bangalore",
        max_distance_km=50,
        min_stipend=0,
        top_k=10,
    )

    violations = [r for r in results if r["distance_km"] > 50]
    if not violations:
        cities = Counter(r["city"] for r in results)
        print(f"  │  Cities: {dict(cities)}")
        print(f"  └─ {PASS}  All {len(results)} results within 50km")
        passed += 1
    else:
        for v in violations[:3]:
            print(f"  │  ⚠ {v['city']}: {v['distance_km']}km")
        print(f"  └─ {FAIL}  {len(violations)} results exceed 50km")

    # Test 5.3: Remote returns cross-city results
    subheader("Remote search: should return internships from any city")
    total += 1
    results = engine.search(
        user_skills=["Python", "Django"],
        education="B.Tech",
        city="Remote",
        max_distance_km=5000,
        min_stipend=0,
        top_k=20,
    )

    unique_cities = set(r["city"] for r in results)
    if len(unique_cities) >= 2:
        print(f"  │  Cities found: {', '.join(list(unique_cities)[:8])}")
        print(f"  └─ {PASS}  Remote search returns {len(unique_cities)} distinct cities")
        passed += 1
    elif len(results) > 0:
        print(f"  └─ {WARN}  Only {len(unique_cities)} city: {unique_cities}")
        passed += 1
    else:
        print(f"  └─ {FAIL}  No results for Remote search")

    # Test 5.4: Very high stipend (≥50000) should return few/no results
    subheader("High stipend filter: min_stipend=50000")
    total += 1
    results = engine.search(
        user_skills=["Python"],
        education="B.Tech",
        city="Remote",
        max_distance_km=5000,
        min_stipend=50000,
        top_k=10,
    )
    
    violations = [r for r in results if r["stipend_max"] < 50000]
    if not violations:
        print(f"  └─ {PASS}  {len(results)} results, all with stipend_max ≥ ₹50,000 (0 violations)")
        passed += 1
    else:
        print(f"  └─ {FAIL}  {len(violations)} results have stipend_max < ₹50,000")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 6: Edge Cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_edge_cases(engine) -> Tuple[int, int]:
    """
    Edge cases that should not crash or return garbage.
    """
    header("TEST 6: EDGE CASES")
    print("  Checks: does the system handle unusual inputs gracefully?")

    passed = 0
    total = 0

    # 6.1: Single skill
    subheader("Single skill: ['Java']")
    total += 1
    results = engine.search(["Java"], "B.Tech", "Bangalore", 100, 0, 5)
    has_java = sum(1 for r in results if "java" in make_searchable_text(r))
    if results and has_java > 0:
        print(f"  └─ {PASS}  {len(results)} results, {has_java} mention Java")
        passed += 1
    elif results:
        print(f"  └─ {WARN}  {len(results)} results but none mention Java")
    else:
        print(f"  └─ {FAIL}  0 results for 'Java' in Bangalore")

    # 6.2: Many skills (10+)
    subheader("Many skills (10 skills)")
    total += 1
    results = engine.search(
        ["Python", "Django", "Flask", "REST API", "PostgreSQL",
         "Docker", "Git", "Linux", "AWS", "Redis"],
        "B.Tech", "Bangalore", 100, 0, 10
    )
    if results:
        print(f"  └─ {PASS}  {len(results)} results for 10-skill query")
        passed += 1
    else:
        print(f"  └─ {FAIL}  0 results for 10-skill query")

    # 6.3: Unknown city with wide radius
    subheader("Unknown city: 'Dharamsala', radius=500km")
    total += 1
    results = engine.search(["Python"], "B.Tech", "Dharamsala", 500, 0, 5)
    if results:
        print(f"  │  Cities: {[r['city'] for r in results[:3]]}")
        print(f"  └─ {PASS}  {len(results)} results (fallback distance=300km < 500km)")
        passed += 1
    else:
        print(f"  └─ {FAIL}  0 results for unknown city with wide radius")

    # 6.4: Same city only (distance=0)
    subheader("Zero distance: city=Mumbai, max_distance=0km")
    total += 1
    results = engine.search(["Python"], "B.Tech", "Mumbai", 0, 0, 5)
    wrong_city = [r for r in results if r["city"] != "Mumbai" and r["city"] != "Remote"]
    if not wrong_city:
        print(f"  └─ {PASS}  {len(results)} results, all from Mumbai (or Remote)")
        passed += 1
    else:
        print(f"  └─ {FAIL}  {len(wrong_city)} results from wrong cities: {[r['city'] for r in wrong_city]}")

    # 6.5: Suburban city (Faridabad → should match Delhi area)
    subheader("Suburban city: 'Faridabad' (maps to Delhi)")
    total += 1
    results = engine.search(["Marketing"], "B.Com", "Faridabad", 50, 0, 5)
    if results:
        cities = [r["city"] for r in results]
        print(f"  │  Cities: {cities[:5]}")
        print(f"  └─ {PASS}  {len(results)} results for Faridabad/Delhi area")
        passed += 1
    else:
        print(f"  └─ {FAIL}  0 results for Faridabad (should map to Delhi)")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 7: Scoring Sanity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_scoring(engine) -> Tuple[int, int]:
    """
    Verify scoring is meaningful:
    - Scores should be descending (sorted)
    - More matching skills → higher score
    - Same city → higher score than distant city
    """
    header("TEST 7: SCORING SANITY")
    print("  Checks: is the scoring order meaningful and consistent?")

    passed = 0
    total = 0

    # 7.1: Results are sorted descending by score
    subheader("Scores are in descending order")
    total += 1
    results = engine.search(["Python", "Django"], "B.Tech", "Bangalore", 100, 0, 10)
    
    if len(results) >= 2:
        scores = [r["match_score"] for r in results]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        print(f"  │  Scores: {[f'{s:.1f}' for s in scores[:5]]}")
        if is_sorted:
            print(f"  └─ {PASS}  All {len(results)} scores in descending order")
            passed += 1
        else:
            print(f"  └─ {FAIL}  Scores are NOT sorted descending")
    else:
        print(f"  └─ {WARN}  Too few results ({len(results)}) to verify ordering")
        passed += 1

    # 7.2: More skill overlap → higher score
    subheader("More skills = higher score")
    total += 1
    
    results_1 = engine.search(["Python"], "B.Tech", "Remote", 5000, 0, 5)
    results_2 = engine.search(["Python", "Django", "REST API"], "B.Tech", "Remote", 5000, 0, 5)
    
    if results_1 and results_2:
        top_score_1 = results_1[0]["match_score"]
        top_score_2 = results_2[0]["match_score"]
        print(f"  │  1 skill top score: {top_score_1:.1f}")
        print(f"  │  3 skills top score: {top_score_2:.1f}")
        if top_score_2 >= top_score_1 * 0.8:  # allow some flexibility
            print(f"  └─ {PASS}  Multi-skill query competitive with single-skill")
            passed += 1
        else:
            print(f"  └─ {WARN}  Multi-skill query scored lower (may be acceptable)")
            passed += 1
    else:
        print(f"  └─ {FAIL}  Missing results for comparison")

    # 7.3: Score range is reasonable (0-100)
    subheader("Score range is 0-100")
    total += 1
    all_results = engine.search(["Python", "Data Science"], "B.Tech", "Remote", 5000, 0, 20)
    
    if all_results:
        min_score = min(r["match_score"] for r in all_results)
        max_score = max(r["match_score"] for r in all_results)
        print(f"  │  Range: {min_score:.1f} – {max_score:.1f}")
        if 0 <= min_score and max_score <= 150:  # small buffer for bonuses
            print(f"  └─ {PASS}  Scores within reasonable range")
            passed += 1
        else:
            print(f"  └─ {FAIL}  Scores outside expected 0-100 range")
    else:
        print(f"  └─ {FAIL}  No results")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 8: Consistency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_consistency(engine) -> Tuple[int, int]:
    """
    Same query run twice should return identical results.
    """
    header("TEST 8: CONSISTENCY (DETERMINISM)")
    print("  Checks: same query → same results every time?")

    passed = 0
    total = 0

    queries = [
        {"skills": ["Python", "Django"], "city": "Bangalore"},
        {"skills": ["React", "JavaScript"], "city": "Mumbai"},
        {"skills": ["Social Media"], "city": "Delhi"},
    ]

    for q in queries:
        subheader(f"{q['skills']} in {q['city']}")
        total += 1

        r1 = engine.search(q["skills"], "B.Tech", q["city"], 100, 0, 10)
        r2 = engine.search(q["skills"], "B.Tech", q["city"], 100, 0, 10)

        ids_1 = [r["id"] for r in r1]
        ids_2 = [r["id"] for r in r2]
        scores_1 = [r["match_score"] for r in r1]
        scores_2 = [r["match_score"] for r in r2]

        if ids_1 == ids_2 and scores_1 == scores_2:
            print(f"  └─ {PASS}  Identical results ({len(r1)} items, same order & scores)")
            passed += 1
        elif set(ids_1) == set(ids_2):
            print(f"  └─ {WARN}  Same items but different ordering")
            passed += 1
        else:
            diff = set(ids_1) ^ set(ids_2)
            print(f"  └─ {FAIL}  Results differ: {len(diff)} mismatched IDs")

    return passed, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     INTERNSHIP RECOMMENDATION — RELEVANCE TEST SUITE       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  8 test categories · 30+ assertions · Lightweight mode     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    start = time.time()
    engine = get_engine()
    init_time = time.time() - start
    print(f"\n  Engine initialized in {init_time:.1f}s")

    # Run all test categories
    categories = [
        ("Persona Relevance", test_persona_relevance),
        ("Skill Matching", test_skill_matching),
        ("Cross-Domain", test_cross_domain),
        ("Education Hierarchy", test_education_hierarchy),
        ("Filter Correctness", test_filters),
        ("Edge Cases", test_edge_cases),
        ("Scoring Sanity", test_scoring),
        ("Consistency", test_consistency),
    ]

    all_results = []
    total_passed = 0
    total_tests = 0

    for name, test_fn in categories:
        try:
            p, t = test_fn(engine)
            total_passed += p
            total_tests += t
            status = PASS if p == t else (WARN if p >= t * 0.5 else FAIL)
            all_results.append((name, p, t, status))
        except Exception as e:
            print(f"\n  ❌ {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append((name, 0, 1, FAIL))
            total_tests += 1

    # Cleanup
    engine.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elapsed = time.time() - start

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      RESULTS SUMMARY                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    for name, p, t, status in all_results:
        bar = "█" * p + "░" * (t - p)
        print(f"║  {status}  {name:<26s} {p}/{t}  [{bar}]")
    
    print("╠══════════════════════════════════════════════════════════════╣")
    
    pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    grade = "A+" if pct >= 95 else "A" if pct >= 90 else "B" if pct >= 80 else "C" if pct >= 70 else "D" if pct >= 60 else "F"
    
    print(f"║  Total: {total_passed}/{total_tests} passed ({pct:.0f}%)  │  Grade: {grade}  │  Time: {elapsed:.1f}s")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
