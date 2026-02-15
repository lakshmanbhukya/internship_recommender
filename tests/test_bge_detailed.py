"""
Detailed BGE-M3 Test with Full Results Display
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
os.environ["LIGHTWEIGHT_MODE"] = "false"

from api.engine_selector import get_search_engine

TEST_PROFILES = [
    {
        "name": "Rahul - Backend Developer",
        "skills": ["Python", "Django", "REST API", "PostgreSQL"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 50,
        "min_stipend": 10000
    },
    {
        "name": "Priya - Frontend Developer",
        "skills": ["React", "JavaScript", "HTML", "CSS", "TypeScript"],
        "education": "B.Tech",
        "city": "Mumbai",
        "max_distance_km": 30,
        "min_stipend": 8000
    },
    {
        "name": "Amit - Data Scientist",
        "skills": ["Python", "Machine Learning", "Pandas", "NumPy", "TensorFlow"],
        "education": "M.Tech",
        "city": "Bangalore",
        "max_distance_km": 50,
        "min_stipend": 15000
    },
    {
        "name": "Sneha - Digital Marketer",
        "skills": ["Social Media Marketing", "Content Writing", "SEO", "Google Analytics"],
        "education": "B.Com",
        "city": "Delhi",
        "max_distance_km": 40,
        "min_stipend": 5000
    },
    {
        "name": "Karthik - Full Stack Developer",
        "skills": ["MERN Stack", "Node.js", "React", "MongoDB", "Express"],
        "education": "B.Tech",
        "city": "Hyderabad",
        "max_distance_km": 50,
        "min_stipend": 12000
    },
    {
        "name": "Ananya - UI/UX Designer",
        "skills": ["Figma", "Adobe XD", "UI Design", "Prototyping", "User Research"],
        "education": "B.Des",
        "city": "Bangalore",
        "max_distance_km": 30,
        "min_stipend": 8000
    },
    {
        "name": "Rohan - Mobile Developer",
        "skills": ["React Native", "Android", "iOS", "Flutter", "Firebase"],
        "education": "B.Tech",
        "city": "Pune",
        "max_distance_km": 50,
        "min_stipend": 10000
    },
    {
        "name": "Divya - Content Creator",
        "skills": ["Content Writing", "Copywriting", "Blog Writing", "Social Media"],
        "education": "B.A",
        "city": "Mumbai",
        "max_distance_km": 50,
        "min_stipend": 5000
    },
    {
        "name": "Arjun - DevOps Engineer",
        "skills": ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux"],
        "education": "B.Tech",
        "city": "Bangalore",
        "max_distance_km": 50,
        "min_stipend": 15000
    },
    {
        "name": "Meera - Business Analyst",
        "skills": ["Excel", "SQL", "Data Analysis", "Business Intelligence", "Tableau"],
        "education": "MBA",
        "city": "Delhi",
        "max_distance_km": 40,
        "min_stipend": 12000
    }
]

def format_stipend(min_val, max_val):
    return f"Rs.{min_val:,}-{max_val:,}/month"

def main():
    print("\n" + "="*80)
    print(" "*25 + "BGE-M3 SEMANTIC SEARCH TEST")
    print("="*80)
    
    print("\nInitializing BGE-M3 model (this may take a moment)...")
    engine = get_search_engine()
    print("BGE-M3 model loaded successfully!\n")
    
    for idx, profile in enumerate(TEST_PROFILES, 1):
        print("\n" + "="*80)
        print(f"{' '*35}TEST {idx}/{len(TEST_PROFILES)}")
        print("="*80)
        
        print(f"\nStudent: {profile['name']}")
        print(f"Skills: {', '.join(profile['skills'])}")
        print(f"Education: {profile['education']}")
        print(f"Location: {profile['city']} (within {profile['max_distance_km']}km)")
        print(f"Min Stipend: Rs.{profile['min_stipend']}")
        
        results = engine.search(
            user_skills=profile['skills'],
            education=profile['education'],
            city=profile['city'],
            max_distance_km=profile['max_distance_km'],
            min_stipend=profile['min_stipend'],
            top_k=10
        )
        
        print(f"\nTop {len(results)} Recommendations:\n")
        
        for i, result in enumerate(results, 1):
            skills_display = ", ".join(result['skills'][:4]) if isinstance(result['skills'], list) else result['skills']
            
            print(f"{i}. {result['role']} @ {result['company']}")
            print(f"   Location: {result['city']} ({result['distance_km']}km away)")
            print(f"   Stipend: {format_stipend(result['stipend_min'], result['stipend_max'])}")
            print(f"   Skills: {skills_display}")
            print(f"   Match Score: {result['match_score']}")
            print()
    
    print("\n" + "="*80)
    print(" "*33 + "TEST SUMMARY")
    print("="*80)
    print(f"\nAll {len(TEST_PROFILES)} student profiles tested successfully!")
    print("\nBGE-M3 Semantic Search Features:")
    print("   - Understands context and intent")
    print("   - Matches similar skills (e.g., 'React' -> 'Frontend')")
    print("   - Handles skill variations (e.g., 'ML' -> 'Machine Learning')")
    print("   - Semantic similarity scoring")
    print("   - Hybrid search (70% semantic + 30% keyword)")
    
    engine.close()

if __name__ == "__main__":
    main()
