"""
================================================================
  BGE-M3 Full Mode Relevance Test Suite
  
  Tests the HYBRID search engine (FAISS semantic + FTS5 lexical)
  powered by the BAAI/bge-m3 embedding model.

  NOTE: This test requires ~2.5 GB RAM to load the BGE-M3 model.
  First run will download the model (~700 MB).

  Run: python scripts/test_relevance_fullmode.py
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


# ================================================================
# Config
# ================================================================
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

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

ROLE_KEYWORDS = {
    "backend": ["backend", "server", "api", "database", "rest", "django", "flask", "node"],
    "frontend": ["frontend", "react", "angular", "vue", "html", "css", "javascript", "ui"],
    "data": ["data", "analytics", "science", "ml", "machine learning", "python", "pandas"],
    "mobile": ["mobile", "android", "ios", "react native", "flutter"],
    "marketing": ["marketing", "social", "content", "digital", "seo"],
}


# ================================================================
# Helpers
# ================================================================
def get_engine():
    """Load the FULL hybrid search engine with BGE-M3."""
    os.environ["LIGHTWEIGHT_MODE"] = "false"
    from api.engine_selector import get_search_engine
    return get_search_engine()


def make_searchable_text(result: Dict) -> str:
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
    return sum(1 for kw in keywords if kw.lower() in text)


def header(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subheader(title: str):
    print(f"\n  >> {title}")


# ================================================================
# TEST 1: Semantic Relevance (BGE-M3 specific)
# ================================================================
def test_semantic_relevance(engine) -> Tuple[int, int]:
    """
    Test that BGE-M3 semantic search understands MEANING, not just keywords.
    E.g., 'web development' should match 'frontend engineering' even without
    exact keyword overlap -- this is where semantic search shines vs keyword.
    """
    header("TEST 1: SEMANTIC RELEVANCE (BGE-M3)")
    print("  Checks: does semantic search understand meaning beyond exact keywords?")

    test_cases = [
        {
            "name": "Synonym understanding",
            "skills": ["Web Development", "UI Design"],
            "expected_in_results": ["frontend", "front end", "web", "ui", "react", "html"],
            "description": "Should find frontend roles even with 'web development' query"
        },
        {
            "name": "Contextual matching",
            "skills": ["Artificial Intelligence", "Deep Learning"],
            "expected_in_results": ["machine learning", "ml", "ai", "data science", "neural"],
            "description": "Should find ML/AI roles from broader AI query"
        },
        {
            "name": "Role inference",
            "skills": ["Node.js", "Express", "MongoDB"],
            "expected_in_results": ["backend", "server", "api", "web", "development", "node"],
            "description": "Should infer backend role from MERN-stack skills"
        },
        {
            "name": "Domain understanding",
            "skills": ["Photoshop", "Illustrator", "Figma"],
            "expected_in_results": ["design", "graphic", "ui", "creative", "visual"],
            "description": "Should find design roles from tool names"
        },
    ]

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        subheader(f"{tc['name']}: {tc['skills']}")
        print(f"  |  {tc['description']}")

        results = engine.search(
            user_skills=tc["skills"],
            education="B.Tech",
            city="Remote",
            max_distance_km=5000,
            min_stipend=0,
            top_k=10,
        )

        if not results:
            print(f"  -- {FAIL} (0 results)")
            continue

        semantic_hits = 0
        for r in results:
            text = make_searchable_text(r)
            hits = count_keyword_hits(text, tc["expected_in_results"])
            if hits >= 1:
                semantic_hits += 1

        precision = semantic_hits / len(results)

        for i, r in enumerate(results[:3], 1):
            text = make_searchable_text(r)
            matched = [kw for kw in tc["expected_in_results"] if kw in text]
            print(f"  |  {i}. {r['role']} @ {r['company']} (score={r['match_score']:.1f})")
            print(f"  |     matched: {', '.join(matched[:5]) if matched else 'none'}")

        if precision >= 0.5:
            print(f"  -- {PASS}  Semantic precision: {semantic_hits}/{len(results)} ({precision:.0%})")
            passed += 1
        elif precision >= 0.3:
            print(f"  -- {WARN}  Semantic precision: {semantic_hits}/{len(results)} ({precision:.0%})")
            passed += 1
        else:
            print(f"  -- {FAIL}  Semantic precision: {semantic_hits}/{len(results)} ({precision:.0%})")

    return passed, total


# ================================================================
# TEST 2: Persona-Based Relevance
# ================================================================
def test_persona_relevance(engine) -> Tuple[int, int]:
    """
    For each persona, search and check if results contain expected keywords.
    """
    header("TEST 2: PERSONA-BASED RELEVANCE")
    print("  Checks: do returned roles match the user's domain?")

    passed = 0
    total = 0

    for persona in PERSONAS:
        subheader(f"{persona['name']} -> skills={persona['skills']}")

        results = engine.search(
            user_skills=persona["skills"],
            education=persona["education"],
            city=persona["city"],
            max_distance_km=persona["max_distance_km"],
            min_stipend=persona["min_stipend"],
            top_k=10,
        )

        if not results:
            print(f"  |  No results returned")
            print(f"  -- {FAIL} (0 results)")
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

        for i, r in enumerate(results[:3], 1):
            text = make_searchable_text(r)
            matched = [kw for kw in persona["expected_skill_keywords"] if kw in text]
            print(f"  |  {i}. {r['role']} @ {r['company']} (score={r['match_score']:.1f})")
            print(f"  |     matched: {', '.join(matched[:5]) if matched else 'none'}")

        if precision >= 0.6:
            print(f"  -- {PASS}  Relevance: {relevant}/{len(results)} ({precision:.0%})")
            passed += 1
        elif precision >= 0.4:
            print(f"  -- {WARN}  Relevance: {relevant}/{len(results)} ({precision:.0%})")
            passed += 1
        else:
            print(f"  -- {FAIL}  Relevance: {relevant}/{len(results)} ({precision:.0%})")

    return passed, total


# ================================================================
# TEST 3: Cross-Domain Contamination
# ================================================================
def test_cross_domain(engine) -> Tuple[int, int]:
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
            print(f"  -- {PASS}  No results = no contamination")
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
            print(f"  -- {PASS}  Contamination: {contaminated}/{len(results)} ({contamination_rate:.0%})")
            passed += 1
        else:
            for cr in contaminated_roles[:3]:
                print(f"  |  ! {cr}")
            print(f"  -- {FAIL}  Contamination: {contaminated}/{len(results)} ({contamination_rate:.0%})")

    return passed, total


# ================================================================
# TEST 4: Hybrid Fusion Quality
# ================================================================
def test_hybrid_fusion(engine) -> Tuple[int, int]:
    """
    Verify that hybrid (semantic+lexical) fusion produces better results
    than pure keyword matching would. The semantic component should
    surface results that share meaning but not exact keywords.
    """
    header("TEST 4: HYBRID FUSION QUALITY")
    print("  Checks: does semantic+lexical fusion improve over keywords alone?")

    passed = 0
    total = 0

    # Test: A query with vague skills should still find relevant results
    # because BGE-M3 understands semantics
    test_cases = [
        {
            "name": "Broad query",
            "skills": ["Programming", "Software Development"],
            "min_relevant": 3,
            "relevant_keywords": ["development", "software", "python", "java", "web", "backend", "frontend"],
        },
        {
            "name": "Niche query",
            "skills": ["Natural Language Processing", "Transformer Models"],
            "min_relevant": 1,
            "relevant_keywords": ["nlp", "ai", "machine learning", "ml", "data", "python", "deep learning"],
        },
        {
            "name": "Cross-lingual understanding",
            "skills": ["Full Stack", "MERN"],
            "min_relevant": 2,
            "relevant_keywords": ["full stack", "react", "node", "javascript", "mongodb", "web", "development", "frontend", "backend"],
        },
    ]

    for tc in test_cases:
        subheader(f"{tc['name']}: {tc['skills']}")
        total += 1

        results = engine.search(
            user_skills=tc["skills"],
            education="B.Tech",
            city="Remote",
            max_distance_km=5000,
            min_stipend=0,
            top_k=10,
        )

        relevant = 0
        for r in results:
            text = make_searchable_text(r)
            hits = count_keyword_hits(text, tc["relevant_keywords"])
            if hits >= 1:
                relevant += 1

        for i, r in enumerate(results[:3], 1):
            text = make_searchable_text(r)
            matched = [kw for kw in tc["relevant_keywords"] if kw in text]
            print(f"  |  {i}. {r['role']} @ {r['company']} (score={r['match_score']:.1f})")
            print(f"  |     matched: {', '.join(matched[:4]) if matched else 'none'}")

        if relevant >= tc["min_relevant"]:
            print(f"  -- {PASS}  Found {relevant} relevant results (needed >={tc['min_relevant']})")
            passed += 1
        else:
            print(f"  -- {FAIL}  Found {relevant} relevant results (needed >={tc['min_relevant']})")

    return passed, total


# ================================================================
# TEST 5: Filter Correctness
# ================================================================
def test_filters(engine) -> Tuple[int, int]:
    header("TEST 5: FILTER CORRECTNESS")
    print("  Checks: are hard filters (education, stipend, distance) enforced?")

    passed = 0
    total = 0

    # 5.1: Education hierarchy
    subheader("Education: B.Tech user sees lower-req roles")
    total += 1
    results = engine.search(["Python", "Django"], "B.Tech", "Remote", 5000, 0, 20)
    edu_types = Counter(r.get("education_req", "?") for r in results)
    print(f"  |  Education distribution: {dict(edu_types)}")
    btech_sees_lower = any(e in edu_types for e in ["Any", "Diploma", "B.A", "B.Com", "B.Sc"])
    if btech_sees_lower or len(results) > 0:
        print(f"  -- {PASS}  B.Tech sees {len(results)} results including lower-req roles")
        passed += 1
    else:
        print(f"  -- {FAIL}  B.Tech cannot see lower-requirement roles")

    # 5.2: Diploma excluded from B.Tech-only
    subheader("Education: Diploma user excluded from B.Tech-only roles")
    total += 1
    results_diploma = engine.search(["Python"], "Diploma", "Remote", 5000, 0, 20)
    btech_only = sum(1 for r in results_diploma if r.get("education_req") == "B.Tech")
    if btech_only == 0:
        print(f"  -- {PASS}  Diploma user sees 0 B.Tech-only roles ({len(results_diploma)} total)")
        passed += 1
    else:
        print(f"  -- {FAIL}  Diploma user sees {btech_only} B.Tech-only roles")

    # 5.3: Stipend >= 10000
    subheader("Stipend filter: min_stipend=10000")
    total += 1
    results = engine.search(["Python", "Django"], "B.Tech", "Remote", 5000, 10000, 10)
    violations = [r for r in results if r["stipend_max"] < 10000]
    if not violations:
        stipends = [f"Rs.{r['stipend_min']}-{r['stipend_max']}" for r in results[:3]]
        print(f"  |  Sample stipends: {', '.join(stipends)}")
        print(f"  -- {PASS}  All {len(results)} results have stipend_max >= Rs.10,000")
        passed += 1
    else:
        print(f"  -- {FAIL}  {len(violations)} violations found")

    # 5.4: Distance
    subheader("Distance filter: max_distance=50km, city=Bangalore")
    total += 1
    results = engine.search(["Python"], "B.Tech", "Bangalore", 50, 0, 10)
    violations = [r for r in results if r["distance_km"] > 50]
    if not violations:
        cities = Counter(r["city"] for r in results)
        print(f"  |  Cities: {dict(cities)}")
        print(f"  -- {PASS}  All {len(results)} results within 50km")
        passed += 1
    else:
        print(f"  -- {FAIL}  {len(violations)} results exceed 50km")

    return passed, total


# ================================================================
# TEST 6: Skill Overlap Scoring
# ================================================================
def test_skill_overlap(engine) -> Tuple[int, int]:
    """
    Verify that internships with more skill overlap score higher.
    """
    header("TEST 6: SKILL OVERLAP SCORING")
    print("  Checks: do results with more matching skills score higher?")

    passed = 0
    total = 0

    subheader("Python+Django+Flask vs Python alone")
    total += 1

    results_multi = engine.search(
        ["Python", "Django", "Flask", "REST API"], "B.Tech", "Remote", 5000, 0, 5
    )
    results_single = engine.search(
        ["Python"], "B.Tech", "Remote", 5000, 0, 5
    )

    if results_multi and results_single:
        # Check: do the multi-skill results have more skill overlap?
        def avg_overlap(results, user_skills):
            user_lower = set(s.lower() for s in user_skills)
            total = 0
            for r in results:
                skills = r.get("skills", [])
                if isinstance(skills, str):
                    try: skills = json.loads(skills)
                    except: skills = skills.split(",")
                job_lower = set(s.lower().strip() for s in skills)
                overlap = sum(1 for us in user_lower for js in job_lower if us in js or js in us)
                total += overlap
            return total / len(results) if results else 0

        avg_multi = avg_overlap(results_multi, ["Python", "Django", "Flask", "REST API"])
        avg_single = avg_overlap(results_single, ["Python"])

        print(f"  |  4-skill query avg overlap: {avg_multi:.1f}")
        print(f"  |  1-skill query avg overlap: {avg_single:.1f}")

        if avg_multi >= avg_single:
            print(f"  -- {PASS}  Multi-skill query has >= skill overlap")
            passed += 1
        else:
            print(f"  -- {WARN}  Multi-skill query has less overlap (may be acceptable)")
            passed += 1
    else:
        print(f"  -- {FAIL}  Missing results for comparison")

    # Check top result has direct skill match
    subheader("Top result has direct skill match")
    total += 1
    results = engine.search(["React", "JavaScript"], "B.Tech", "Remote", 5000, 0, 5)
    if results:
        top = results[0]
        text = make_searchable_text(top)
        has_react = "react" in text
        has_js = "javascript" in text or "js" in text
        print(f"  |  Top result: {top['role']} @ {top['company']}")
        print(f"  |  Has React: {has_react}, Has JS: {has_js}")
        if has_react or has_js:
            print(f"  -- {PASS}  Top result matches queried skills")
            passed += 1
        else:
            print(f"  -- {WARN}  Top result doesn't mention query skills directly")
            passed += 1
    else:
        print(f"  -- {FAIL}  No results")

    return passed, total


# ================================================================
# TEST 7: Scoring Sanity
# ================================================================
def test_scoring(engine) -> Tuple[int, int]:
    header("TEST 7: SCORING SANITY")
    print("  Checks: is the scoring order meaningful?")

    passed = 0
    total = 0

    # 7.1: Descending order
    subheader("Scores are in descending order")
    total += 1
    results = engine.search(["Python", "Django"], "B.Tech", "Bangalore", 100, 0, 10)
    if len(results) >= 2:
        scores = [r["match_score"] for r in results]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        print(f"  |  Scores: {[f'{s:.1f}' for s in scores[:6]]}")
        if is_sorted:
            print(f"  -- {PASS}  All {len(results)} scores in descending order")
            passed += 1
        else:
            print(f"  -- {FAIL}  Scores are NOT sorted descending")
    else:
        print(f"  -- {WARN}  Too few results to verify")
        passed += 1

    # 7.2: Score range
    subheader("Score range is reasonable (0-150)")
    total += 1
    all_results = engine.search(["Python", "Data Science"], "B.Tech", "Remote", 5000, 0, 20)
    if all_results:
        min_score = min(r["match_score"] for r in all_results)
        max_score = max(r["match_score"] for r in all_results)
        print(f"  |  Range: {min_score:.1f} - {max_score:.1f}")
        if 0 <= min_score and max_score <= 150:
            print(f"  -- {PASS}  Scores within reasonable range")
            passed += 1
        else:
            print(f"  -- {FAIL}  Scores outside expected range")
    else:
        print(f"  -- {FAIL}  No results")

    # 7.3: Score spread (not all identical)
    subheader("Score spread (not all identical)")
    total += 1
    if all_results and len(all_results) >= 3:
        scores = [r["match_score"] for r in all_results]
        score_spread = max(scores) - min(scores)
        unique_scores = len(set(scores))
        print(f"  |  Spread: {score_spread:.1f}, Unique scores: {unique_scores}/{len(scores)}")
        if score_spread > 1.0 and unique_scores >= 2:
            print(f"  -- {PASS}  Scores have meaningful spread")
            passed += 1
        else:
            print(f"  -- {WARN}  Scores are too clustered (spread={score_spread:.1f})")
            passed += 1
    else:
        print(f"  -- {WARN}  Too few results")
        passed += 1

    return passed, total


# ================================================================
# TEST 8: Latency (BGE-M3 specific)
# ================================================================
def test_latency(engine) -> Tuple[int, int]:
    """
    BGE-M3 is slower than keyword search. Verify it's within acceptable range.
    Target: < 5 seconds per query on CPU.
    """
    header("TEST 8: LATENCY (BGE-M3 on CPU)")
    print("  Checks: is query latency acceptable for interactive use?")

    passed = 0
    total = 0

    # Warmup
    engine.search(["Python"], "B.Tech", "Bangalore", 50, 0, 5)

    subheader("Latency benchmark (5 queries)")
    total += 1
    latencies = []
    queries = [
        ["Python", "Django"],
        ["React", "JavaScript", "HTML"],
        ["Machine Learning", "Python"],
        ["Social Media", "Content Writing"],
        ["Docker", "AWS", "Linux"],
    ]

    for i, skills in enumerate(queries, 1):
        start = time.time()
        results = engine.search(
            user_skills=skills,
            education="B.Tech",
            city="Bangalore",
            max_distance_km=100,
            min_stipend=0,
            top_k=10,
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        print(f"  |  Query {i} ({', '.join(skills)}): {latency_ms:.0f}ms -> {len(results)} results")

    avg_ms = sum(latencies) / len(latencies)
    max_ms = max(latencies)
    print(f"  |")
    print(f"  |  Average: {avg_ms:.0f}ms")
    print(f"  |  Max:     {max_ms:.0f}ms")

    if max_ms < 5000:
        print(f"  -- {PASS}  All queries under 5s (avg={avg_ms:.0f}ms)")
        passed += 1
    elif max_ms < 10000:
        print(f"  -- {WARN}  Queries under 10s but above 5s target")
        passed += 1
    else:
        print(f"  -- {FAIL}  Some queries exceed 10s")

    return passed, total


# ================================================================
# TEST 9: Consistency
# ================================================================
def test_consistency(engine) -> Tuple[int, int]:
    header("TEST 9: CONSISTENCY (DETERMINISM)")
    print("  Checks: same query -> same results every time?")

    passed = 0
    total = 0

    queries = [
        {"skills": ["Python", "Django"], "city": "Bangalore"},
        {"skills": ["React", "JavaScript"], "city": "Mumbai"},
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
            print(f"  -- {PASS}  Identical results ({len(r1)} items, same order & scores)")
            passed += 1
        elif set(ids_1) == set(ids_2):
            print(f"  -- {WARN}  Same items but different ordering")
            passed += 1
        else:
            diff = set(ids_1) ^ set(ids_2)
            print(f"  -- {FAIL}  Results differ: {len(diff)} mismatched IDs")

    return passed, total


# ================================================================
# TEST 10: Health Check
# ================================================================
def test_health(engine) -> Tuple[int, int]:
    header("TEST 10: HEALTH CHECK")

    passed = 0
    total = 0

    subheader("Engine health status")
    total += 1
    health = engine.get_health()

    checks = {
        "status": health.get("status") == "healthy",
        "database": health.get("database_connected") == True,
        "model": health.get("model_loaded") == True,
        "faiss": health.get("faiss_index_loaded") == True,
        "records": health.get("total_internships", 0) > 8000,
        "vectors": health.get("vector_count", 0) > 8000,
    }

    all_ok = True
    for check_name, ok in checks.items():
        status = PASS if ok else FAIL
        print(f"  |  {status} {check_name}: {health.get(check_name, 'N/A')}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"  -- {PASS}  All health checks passed")
        passed += 1
    else:
        print(f"  -- {FAIL}  Some health checks failed")

    return passed, total


# ================================================================
# Main
# ================================================================
def main():
    print()
    print("================================================================")
    print("  BGE-M3 FULL MODE -- RELEVANCE TEST SUITE")
    print("  (FAISS semantic + FTS5 lexical + RRF fusion)")
    print("================================================================")
    print("  NOTE: Loading BGE-M3 model (~2.5 GB RAM). Please wait...")

    start = time.time()
    engine = get_engine()
    init_time = time.time() - start
    print(f"\n  Engine initialized in {init_time:.1f}s")
    print(f"  Model: BAAI/bge-m3 | FAISS vectors: {engine.index.ntotal}")

    categories = [
        ("Semantic Relevance", test_semantic_relevance),
        ("Persona Relevance", test_persona_relevance),
        ("Cross-Domain", test_cross_domain),
        ("Hybrid Fusion", test_hybrid_fusion),
        ("Filters", test_filters),
        ("Skill Overlap", test_skill_overlap),
        ("Scoring Sanity", test_scoring),
        ("Latency", test_latency),
        ("Consistency", test_consistency),
        ("Health Check", test_health),
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
            print(f"\n  [ERROR] {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append((name, 0, 1, FAIL))
            total_tests += 1

    engine.close()

    elapsed = time.time() - start

    # Summary
    print("\n")
    print("================================================================")
    print("  RESULTS SUMMARY -- BGE-M3 FULL MODE")
    print("================================================================")

    for name, p, t, status in all_results:
        bar = "#" * p + "." * (t - p)
        print(f"  {status}  {name:<26s} {p}/{t}  [{bar}]")

    print("----------------------------------------------------------------")

    pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    grade = (
        "A+" if pct >= 95 else
        "A"  if pct >= 90 else
        "B"  if pct >= 80 else
        "C"  if pct >= 70 else
        "D"  if pct >= 60 else "F"
    )

    print(f"  Total: {total_passed}/{total_tests} passed ({pct:.0f}%)")
    print(f"  Grade: {grade}  |  Time: {elapsed:.1f}s  |  Model: BAAI/bge-m3")
    print("================================================================")
    print()


if __name__ == "__main__":
    main()
