"""
Lightweight search engine for low-memory environments (512 MB RAM)
Uses pre-computed embeddings only, no model loading
"""
import sqlite3
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, DATA_DIR

try:
    import faiss
except ImportError:
    faiss = None

from api.utils import get_city_distance

# Skill synonyms for better matching
SKILL_SYNONYMS = {
    "python": ["python", "python3", "py", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "react", "vue", "angular", "node", "nodejs"],
    "java": ["java", "spring", "springboot", "hibernate"],
    "ml": ["machine learning", "ml", "ai", "artificial intelligence", "deep learning", "neural network"],
    "data science": ["data science", "data analysis", "analytics", "ml", "machine learning"],
    "backend": ["backend", "server-side", "api", "rest", "database"],
    "frontend": ["frontend", "ui", "ux", "web", "html", "css"],
    "mobile": ["mobile", "android", "ios", "react native", "flutter"],
    "devops": ["devops", "docker", "kubernetes", "ci/cd", "aws", "cloud"],
    "marketing": ["marketing", "digital marketing", "social media", "seo", "content"],
}

class LightweightSearchEngine:
    """Memory-efficient search: FAISS + keyword matching (no model)"""
    
    def __init__(self):
        self.db_path = str(DB_PATH)
        self.faiss_index_path = str(DATA_DIR / "faiss_index.bin")
        self.id_mapping_path = str(DATA_DIR / "id_mapping.json")
        
        # Load FAISS index
        if faiss:
            print("Loading FAISS index...")
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"[OK] FAISS index loaded: {self.index.ntotal} vectors")
        else:
            self.index = None
            print("[WARNING] FAISS not available")
        
        # Load ID mapping
        with open(self.id_mapping_path, 'r') as f:
            self.id_mapping = json.load(f)['ids']
        print(f"[OK] ID mapping loaded: {len(self.id_mapping)} IDs")
        
        # Connect to SQLite
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        print(f"[OK] Connected to SQLite DB")
        
        # NO MODEL LOADING - saves 1.8 GB RAM
        self.model = None
        print("[OK] Lightweight mode - no model loaded")
    
    def search(self,
               user_skills: List[str],
               education: str,
               city: str,
               max_distance_km: int = 50,
               min_stipend: int = 0,
               top_k: int = 10) -> List[Dict]:
        """Search using keyword matching + filters only"""
        
        # Use keyword search (no semantic search without model)
        query = " ".join(user_skills)
        candidates = self._keyword_search(query, education, min_stipend, city, max_distance_km, top_k * 3)
        
        # Apply scoring
        results = []
        for candidate in candidates[:top_k]:
            results.append({
                "id": candidate['id'],
                "role": candidate['profile'],
                "company": candidate['company'],
                "city": candidate['city'],
                "stipend_min": candidate['stipend_min'],
                "stipend_max": candidate['stipend_max'],
                "duration_months": candidate['duration_months'],
                "education_req": candidate['education_req'],
                "skills": candidate['skills'],
                "perks": candidate['perks'],
                "apply_by": candidate['apply_by'],
                "match_score": candidate['match_score'],
                "distance_km": candidate['distance_km'],
                "freshness_score": candidate['freshness_score']
            })
        
        return results
    
    def _keyword_search(self, query: str, education: str, min_stipend: int, 
                       user_city: str, max_distance_km: int, top_k: int) -> List[Dict]:
        """Keyword-based search with filters and synonyms"""
        # Expand query with synonyms
        keywords = set(query.lower().split())
        expanded_keywords = set()
        for kw in keywords:
            expanded_keywords.add(kw)
            # Add synonyms
            for base, synonyms in SKILL_SYNONYMS.items():
                if kw in synonyms or kw == base:
                    expanded_keywords.update(synonyms)
                    break
        
        # Query with filters
        cursor = self.conn.execute("""
            SELECT id, profile, company, location_normalized,
                   stipend_min, stipend_max, duration_months,
                   education_req, skills, perks, apply_by, freshness_score
            FROM internships
            WHERE education_req IN (?, 'Any')
              AND stipend_min >= ?
            ORDER BY freshness_score DESC
            LIMIT ?
        """, (education, min_stipend, top_k * 5))
        
        results = []
        for row in cursor.fetchall():
            internship_id, profile, company, city, stipend_min, stipend_max, \
            duration_months, education_req, skills_json, perks, apply_by, freshness = row
            
            # Distance filter
            distance_km = get_city_distance(user_city, city)
            if distance_km > max_distance_km:
                continue
            
            # Parse skills properly
            try:
                job_skills = json.loads(skills_json) if skills_json else []
                job_skills = [s.lower().strip() for s in job_skills]
            except:
                job_skills = [s.lower().strip() for s in skills_json.split(",")] if skills_json else []
            
            # Calculate keyword match with expanded keywords
            text = f"{profile} {' '.join(job_skills)}".lower()
            matches = sum(1 for kw in expanded_keywords if kw in text)
            
            # Calculate skill overlap
            skill_overlap = len(set(expanded_keywords) & set(job_skills))
            
            # Combined keyword score
            keyword_score = max(
                matches / len(keywords) if keywords else 0,
                skill_overlap / len(keywords) if keywords else 0
            )
            
            # Filter out zero matches
            if keyword_score == 0:
                continue
            
            # Improved scoring: keyword 70%, freshness 20%, distance 10%
            distance_factor = max(0.5, 1.0 - (distance_km / max_distance_km)) if max_distance_km > 0 else 1.0
            match_score = (keyword_score * 70 + freshness * 20 + distance_factor * 10)
            
            results.append({
                'id': internship_id,
                'profile': profile,
                'company': company,
                'city': city,
                'stipend_min': stipend_min,
                'stipend_max': stipend_max,
                'duration_months': duration_months,
                'education_req': education_req,
                'skills': job_skills,
                'perks': perks,
                'apply_by': apply_by,
                'match_score': round(match_score, 1),
                'distance_km': round(distance_km, 1),
                'freshness_score': freshness
            })
        
        # Sort by match score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results
    
    def get_health(self) -> Dict:
        """Health check"""
        total = self.conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
        return {
            "status": "healthy",
            "mode": "lightweight (512 MB)",
            "database_connected": True,
            "model_loaded": False,
            "faiss_index_loaded": self.index is not None,
            "total_internships": total,
            "search_type": "keyword matching + filters"
        }
    
    def close(self):
        if self.conn:
            self.conn.close()

# Singleton
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = LightweightSearchEngine()
    return _engine
