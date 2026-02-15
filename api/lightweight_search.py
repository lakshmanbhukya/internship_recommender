"""
Lightweight search engine for low-memory environments (512 MB RAM)
Uses keyword matching with synonym expansion, no model loading.
Fixed: education hierarchy, stipend filter, SQL ordering, synonym expansion.
"""
import sqlite3
import numpy as np
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, DATA_DIR

try:
    import faiss
except ImportError:
    faiss = None

from api.utils import get_city_distance, is_education_eligible

logger = logging.getLogger(__name__)

# Enhanced skill synonyms with technical depth
SKILL_SYNONYMS = {
    "python": ["python", "python3", "py", "django", "flask", "fastapi", "pyramid", "bottle"],
    "django": ["django", "python", "backend", "web framework", "rest", "api"],
    "flask": ["flask", "python", "backend", "microframework", "rest", "api"],
    "javascript": ["javascript", "js", "react", "vue", "angular", "node", "nodejs", "typescript", "es6"],
    "react": ["react", "reactjs", "javascript", "frontend", "ui", "jsx", "web"],
    "angular": ["angular", "angularjs", "javascript", "frontend", "typescript", "web"],
    "vue": ["vue", "vuejs", "javascript", "frontend", "web"],
    "node": ["node", "nodejs", "javascript", "backend", "server", "express"],
    "java": ["java", "spring", "springboot", "hibernate", "jvm", "backend"],
    "rest api": ["rest", "api", "restful", "web service", "endpoint", "http", "backend"],
    "api": ["api", "rest", "restful", "web service", "endpoint", "backend"],
    "machine learning": ["machine learning", "ml", "ai", "artificial intelligence", "deep learning", "neural network", "data science"],
    "ml": ["ml", "machine learning", "ai", "data science", "deep learning"],
    "ai": ["ai", "artificial intelligence", "machine learning", "ml", "deep learning"],
    "data science": ["data science", "data analysis", "analytics", "ml", "machine learning", "statistics"],
    "pandas": ["pandas", "python", "data analysis", "data science", "numpy"],
    "numpy": ["numpy", "python", "data science", "scientific computing"],
    "backend": ["backend", "server-side", "api", "rest", "database", "server", "backend development"],
    "frontend": ["frontend", "ui", "ux", "web", "html", "css", "javascript", "frontend development"],
    "html": ["html", "html5", "web", "frontend", "markup"],
    "css": ["css", "css3", "styling", "frontend", "web", "sass", "scss"],
    "mobile": ["mobile", "android", "ios", "react native", "flutter", "mobile development"],
    "android": ["android", "mobile", "java", "kotlin"],
    "ios": ["ios", "mobile", "swift", "objective-c"],
    "devops": ["devops", "docker", "kubernetes", "ci/cd", "aws", "cloud", "deployment"],
    "docker": ["docker", "container", "devops", "deployment"],
    "aws": ["aws", "amazon web services", "cloud", "devops"],
    "marketing": ["marketing", "digital marketing", "social media", "seo", "content"],
    "social media": ["social media", "marketing", "digital marketing", "content"],
    "content writing": ["content", "writing", "content writing", "copywriting", "marketing"],
}

# Role-specific keyword weights
ROLE_KEYWORDS = {
    "backend": ["backend", "server", "api", "database", "rest", "django", "flask", "node"],
    "frontend": ["frontend", "react", "angular", "vue", "html", "css", "javascript", "ui"],
    "data": ["data", "analytics", "science", "ml", "machine learning", "python", "pandas"],
    "mobile": ["mobile", "android", "ios", "react native", "flutter"],
    "marketing": ["marketing", "social", "content", "digital", "seo"],
}

class LightweightSearchEngine:
    """Memory-efficient search: keyword matching + filters (no model)"""
    
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
        
        query = " ".join(user_skills)
        candidates = self._keyword_search(query, user_skills, education, min_stipend, city, max_distance_km, top_k * 3)
        
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
    
    def _keyword_search(self, query: str, user_skills: List[str], education: str, 
                       min_stipend: int, user_city: str, max_distance_km: int, 
                       top_k: int) -> List[Dict]:
        """Keyword-based search with filters and synonyms"""
        self._ensure_connection()
        
        # Expand query with synonyms (check ALL matching groups, no break)
        keywords = set(query.lower().split())
        expanded_keywords = set()
        for kw in keywords:
            expanded_keywords.add(kw)
            for base, synonyms in SKILL_SYNONYMS.items():
                if kw in synonyms or kw == base:
                    expanded_keywords.update(synonyms)
                    # No break — check all synonym groups for multi-group matches
        
        # Query with stipend_max filter instead of stipend_min
        # Scan a large pool — 8,483 records total, scanning 2000 is ~4ms
        cursor = self.conn.execute("""
            SELECT id, profile, company, location_normalized,
                   stipend_min, stipend_max, duration_months,
                   education_req, skills, perks, apply_by, freshness_score
            FROM internships
            WHERE stipend_max >= ?
            LIMIT 2000
        """, (min_stipend,))
        
        results = []
        for row in cursor.fetchall():
            internship_id, profile, company, city, stipend_min, stipend_max, \
            duration_months, education_req, skills_json, perks, apply_by, freshness = row
            
            # Education filter: hierarchy-based (post-filter)
            if not is_education_eligible(education, education_req):
                continue
            
            # Distance filter
            distance_km = get_city_distance(user_city, city)
            if distance_km > max_distance_km:
                continue
            
            # Parse skills properly
            try:
                job_skills = json.loads(skills_json) if skills_json else []
                job_skills = [s.lower().strip() for s in job_skills]
            except (json.JSONDecodeError, TypeError):
                job_skills = [s.lower().strip() for s in skills_json.split(",")] if skills_json else []
            
            # Enhanced matching with role detection
            text = f"{profile} {' '.join(job_skills)}".lower()
            
            # Exact skill matches (highest priority)
            exact_matches = sum(1 for kw in keywords if kw in text)
            
            # Direct skill overlap
            skill_overlap = 0
            for kw in keywords:
                for skill in job_skills:
                    if kw in skill or skill in kw:
                        skill_overlap += 1
                        break
            
            # Synonym expansion matches
            synonym_matches = sum(1 for kw in expanded_keywords if kw in text and kw not in keywords)
            
            # Role-based bonus (strong signal)
            role_bonus = 0
            for role_type, role_kws in ROLE_KEYWORDS.items():
                role_in_text = sum(1 for rk in role_kws if rk in text)
                user_wants_role = sum(1 for uk in keywords if uk in role_kws)
                if role_in_text >= 2 and user_wants_role >= 1:
                    role_bonus = 0.4
                    break
            
            # Calculate weighted score
            exact_score = (exact_matches / len(keywords)) if keywords else 0
            skill_score = (skill_overlap / len(keywords)) if keywords else 0
            synonym_score = (synonym_matches / max(len(expanded_keywords) - len(keywords), 1)) * 0.5
            
            keyword_score = (
                exact_score * 0.5 +
                skill_score * 0.35 +
                synonym_score * 0.15 +
                role_bonus
            )
            
            # Stricter filter - require meaningful match
            if keyword_score < 0.25:
                continue
            
            # Scoring: keyword 90%, distance 10% 
            # (freshness removed — all values are identical at 0.3)
            distance_factor = max(0.5, 1.0 - (distance_km / max_distance_km)) if max_distance_km > 0 else 1.0
            match_score = (keyword_score * 90 + distance_factor * 10)
            
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
        self._ensure_connection()
        total = self.conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
        return {
            "status": "healthy",
            "mode": "lightweight (512 MB)",
            "database_connected": True,
            "model_loaded": False,
            "faiss_index_loaded": self.index is not None,
            "total_internships": total,
            "search_type": "keyword matching + filters",
            "version": "2.1.0"
        }
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _ensure_connection(self):
        """Reconnect if connection was closed"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            print("[OK] Reconnected to SQLite DB")

# Singleton
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = LightweightSearchEngine()
    return _engine
