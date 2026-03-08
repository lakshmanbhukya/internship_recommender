"""
Recommendation Output Comparison Script
========================================
Runs 10 user queries on BOTH Lightweight and Full (BGE-M3) modes.
Top 5 results per query are saved to 'recommendation_output.txt'.

Run:  python scripts/compare_recommendations.py
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

# ---- 10 Diverse User Queries ----
USER_QUERIES = [
    {
        "id": 1,
        "label": "Backend Developer (Bangalore)",
        "skills": ["Python", "Django", "REST API"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 100,
        "min_stipend": 5000,
    },
    {
        "id": 2,
        "label": "Frontend Developer (Mumbai)",
        "skills": ["React", "JavaScript", "HTML", "CSS"],
        "education": "B.Tech",
        "city": "Mumbai",
        "max_distance_km": 100,
        "min_stipend": 0,
    },
    {
        "id": 3,
        "label": "Data Scientist (Remote)",
        "skills": ["Python", "Machine Learning", "Pandas", "NumPy"],
        "education": "B.Tech",
        "city": "Remote",
        "max_distance_km": 5000,
        "min_stipend": 0,
    },
    {
        "id": 4,
        "label": "Marketing Intern (Delhi)",
        "skills": ["Social Media", "Content Writing", "SEO"],
        "education": "B.Com",
        "city": "Delhi",
        "max_distance_km": 50,
        "min_stipend": 0,
    },
    {
        "id": 5,
        "label": "Mobile App Developer (Pune)",
        "skills": ["Android", "Java", "Kotlin"],
        "education": "B.Tech",
        "city": "Pune",
        "max_distance_km": 100,
        "min_stipend": 0,
    },
    {
        "id": 6,
        "label": "DevOps / Cloud (Bangalore, high stipend)",
        "skills": ["Docker", "AWS", "Linux", "CI/CD"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 100,
        "min_stipend": 10000,
    },
    {
        "id": 7,
        "label": "Graphic Designer (Delhi)",
        "skills": ["Photoshop", "Illustrator", "Canva"],
        "education": "B.A",
        "city": "Delhi",
        "max_distance_km": 50,
        "min_stipend": 0,
    },
    {
        "id": 8,
        "label": "Full Stack Developer (Remote)",
        "skills": ["React", "Node.js", "MongoDB", "Express"],
        "education": "B.Tech",
        "city": "Remote",
        "max_distance_km": 5000,
        "min_stipend": 5000,
    },
    {
        "id": 9,
        "label": "Finance / Accounting Intern (Mumbai)",
        "skills": ["Excel", "Tally", "Accounting"],
        "education": "B.Com",
        "city": "Mumbai",
        "max_distance_km": 50,
        "min_stipend": 0,
    },
    {
        "id": 10,
        "label": "Cybersecurity Intern (Hyderabad)",
        "skills": ["Network Security", "Linux", "Python"],
        "education": "B.Tech",
        "city": "Hyderabad",
        "max_distance_km": 100,
        "min_stipend": 0,
    },
]

TOP_K = 5


def format_result(rank, r):
    """Format a single result as readable text."""
    skills = r.get("skills", [])
    if isinstance(skills, str):
        import json
        try:
            skills = json.loads(skills)
        except:
            skills = skills.split(",")
    skills_str = ", ".join(skills[:6])
    if len(skills) > 6:
        skills_str += f" (+{len(skills)-6} more)"

    perks = r.get("perks", "") or ""
    if len(perks) > 80:
        perks = perks[:77] + "..."

    lines = []
    lines.append(f"    #{rank}  {r['role']}")
    lines.append(f"         Company   : {r['company']}")
    lines.append(f"         Location  : {r['city']} ({r['distance_km']:.0f} km)")
    lines.append(f"         Stipend   : Rs.{r['stipend_min']:,} - Rs.{r['stipend_max']:,}/month")
    lines.append(f"         Duration  : {r['duration_months']} months")
    lines.append(f"         Education : {r['education_req']}")
    lines.append(f"         Skills    : {skills_str}")
    lines.append(f"         Score     : {r['match_score']:.1f}/100")
    if perks:
        lines.append(f"         Perks     : {perks}")
    lines.append("")
    return "\n".join(lines)


def run_query(engine, query, mode_label):
    """Run a single query and return formatted text."""
    start = time.time()
    results = engine.search(
        user_skills=query["skills"],
        education=query["education"],
        city=query["city"],
        max_distance_km=query["max_distance_km"],
        min_stipend=query["min_stipend"],
        top_k=TOP_K,
    )
    elapsed = (time.time() - start) * 1000

    lines = []
    lines.append(f"  [{mode_label}] ({elapsed:.0f}ms, {len(results)} results)")
    lines.append("")

    if not results:
        lines.append("    (no results)")
        lines.append("")
    else:
        for i, r in enumerate(results, 1):
            lines.append(format_result(i, r))

    return "\n".join(lines)


def main():
    output_path = Path(__file__).parent.parent / "recommendation_output.txt"
    output_lines = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append("=" * 80)
    output_lines.append("  INTERNSHIP RECOMMENDATION OUTPUT -- SIDE-BY-SIDE COMPARISON")
    output_lines.append(f"  Generated: {timestamp}")
    output_lines.append(f"  Queries: {len(USER_QUERIES)} | Top-K: {TOP_K}")
    output_lines.append(f"  Modes: Lightweight (keyword+FAISS) vs Full (BGE-M3 semantic+FTS5)")
    output_lines.append("=" * 80)
    output_lines.append("")

    # ---- Load Lightweight Engine ----
    print("[1/3] Loading Lightweight engine...")
    os.environ["LIGHTWEIGHT_MODE"] = "true"

    # Force fresh import each time
    import importlib
    if "api.engine_selector" in sys.modules:
        del sys.modules["api.engine_selector"]
    if "api.lightweight_search" in sys.modules:
        del sys.modules["api.lightweight_search"]

    from api.engine_selector import get_search_engine
    engine_light = get_search_engine()

    print("[2/3] Running Lightweight queries...")
    light_results = {}
    for q in USER_QUERIES:
        print(f"  Query {q['id']}: {q['label']}...")
        light_results[q["id"]] = run_query(engine_light, q, "LIGHTWEIGHT")
    engine_light.close()

    # ---- Load Full Engine ----
    print("[3/3] Loading Full BGE-M3 engine (this takes ~60-120s)...")
    os.environ["LIGHTWEIGHT_MODE"] = "false"

    # Force fresh import
    for mod in list(sys.modules.keys()):
        if mod.startswith("api."):
            del sys.modules[mod]

    from api.engine_selector import get_search_engine as get_engine_full
    engine_full = get_engine_full()

    print("  Running Full mode queries...")
    full_results = {}
    for q in USER_QUERIES:
        print(f"  Query {q['id']}: {q['label']}...")
        full_results[q["id"]] = run_query(engine_full, q, "FULL (BGE-M3)")
    engine_full.close()

    # ---- Build Output ----
    for q in USER_QUERIES:
        output_lines.append("-" * 80)
        output_lines.append(f"  QUERY {q['id']}: {q['label']}")
        output_lines.append(f"  Skills    : {', '.join(q['skills'])}")
        output_lines.append(f"  Education : {q['education']}")
        output_lines.append(f"  City      : {q['city']} (max {q['max_distance_km']}km)")
        output_lines.append(f"  Min Stipend: Rs.{q['min_stipend']:,}")
        output_lines.append("-" * 80)
        output_lines.append("")
        output_lines.append(light_results[q["id"]])
        output_lines.append("  " + "- " * 35)
        output_lines.append("")
        output_lines.append(full_results[q["id"]])
        output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("  END OF REPORT")
    output_lines.append("=" * 80)

    # ---- Write to file ----
    final_text = "\n".join(output_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nDone! Output saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
