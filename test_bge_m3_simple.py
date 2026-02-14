"""
BGE-M3 Test - Simplified Version
Saves results to bge_m3_results.txt
"""
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
os.environ["LIGHTWEIGHT_MODE"] = "false"

from api.engine_selector import get_search_engine

PROFILES = [
    {"name": "Backend Dev", "skills": ["Python", "Django", "REST API"], "education": "B.Tech", "city": "Bangalore"},
    {"name": "Frontend Dev", "skills": ["React", "JavaScript", "HTML"], "education": "B.Tech", "city": "Mumbai"},
    {"name": "Data Scientist", "skills": ["Python", "Machine Learning", "Pandas"], "education": "M.Tech", "city": "Bangalore"},
    {"name": "Digital Marketer", "skills": ["Social Media", "Content Writing", "SEO"], "education": "B.Com", "city": "Delhi"},
    {"name": "Full Stack Dev", "skills": ["MERN", "Node.js", "React"], "education": "B.Tech", "city": "Hyderabad"},
    {"name": "UI/UX Designer", "skills": ["Figma", "UI Design", "Prototyping"], "education": "B.Des", "city": "Bangalore"},
    {"name": "Mobile Dev", "skills": ["React Native", "Android", "Flutter"], "education": "B.Tech", "city": "Pune"},
    {"name": "Content Creator", "skills": ["Content Writing", "Copywriting"], "education": "B.A", "city": "Mumbai"},
    {"name": "DevOps Engineer", "skills": ["Docker", "Kubernetes", "AWS"], "education": "B.Tech", "city": "Bangalore"},
    {"name": "Business Analyst", "skills": ["Excel", "SQL", "Tableau"], "education": "MBA", "city": "Delhi"}
]

def main():
    output = []
    output.append("="*80)
    output.append("BGE-M3 SEMANTIC SEARCH TEST RESULTS")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("="*80)
    
    print("Loading BGE-M3 model...")
    engine = get_search_engine()
    print("Model loaded!\n")
    
    for idx, profile in enumerate(PROFILES, 1):
        print(f"Testing {idx}/10: {profile['name']}...")
        
        output.append(f"\n\n{'='*80}")
        output.append(f"TEST {idx}: {profile['name']}")
        output.append(f"{'='*80}")
        output.append(f"Skills: {', '.join(profile['skills'])}")
        output.append(f"Education: {profile['education']}")
        output.append(f"Location: {profile['city']}")
        
        results = engine.search(
            user_skills=profile["skills"],
            education=profile["education"],
            city=profile["city"],
            max_distance_km=50,
            min_stipend=5000,
            top_k=10
        )
        
        output.append(f"\nTop {len(results)} Recommendations:\n")
        
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['role']} @ {r['company']}")
            output.append(f"   Location: {r['city']} ({r['distance_km']}km)")
            output.append(f"   Stipend: Rs.{r['stipend_min']}-{r['stipend_max']}")
            output.append(f"   Skills: {', '.join(r['skills'][:3])}")
            output.append(f"   Match Score: {r['match_score']:.1f}\n")
    
    output.append("\n" + "="*80)
    output.append("TEST COMPLETE")
    output.append("="*80)
    
    # Save to file
    with open("bge_m3_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    
    print("\n✅ Results saved to bge_m3_results.txt")
    engine.close()

if __name__ == "__main__":
    main()
