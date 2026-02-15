"""
Hybrid search engine: FAISS (semantic) + SQLite (lexical) + filters
Fixed: embedding mismatch, RRF fusion, education hierarchy, stipend filter,
       freshness scoring, score clamping, skill overlap, error handling
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

# Check if faiss is installed
try:
    import faiss
except ImportError:
    print("FAISS not installed! Install with: pip install faiss-cpu")
    sys.exit(1)

from sentence_transformers import SentenceTransformer
from api.utils import get_city_distance, is_education_eligible

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Hybrid search: FAISS (semantic) + SQLite (lexical) + filters"""
    
    def __init__(self, 
                 db_path: str = None,
                 faiss_index_path: str = None,
                 id_mapping_path: str = None,
                 model_name: str = "BAAI/bge-m3"):
        
        self.db_path = db_path or str(DB_PATH)
        self.faiss_index_path = faiss_index_path or str(DATA_DIR / "faiss_index.bin")
        self.id_mapping_path = id_mapping_path or str(DATA_DIR / "id_mapping.json")
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(self.faiss_index_path)
        print(f"[OK] FAISS index loaded: {self.index.ntotal} vectors")
        
        # Load ID mapping
        with open(self.id_mapping_path, 'r') as f:
            self.id_mapping = json.load(f)['ids']
        print(f"[OK] ID mapping loaded: {len(self.id_mapping)} IDs")
        
        # Connect to SQLite
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        print(f"[OK] Connected to SQLite DB")
        
        # Load embedding model (CPU) with memory optimization
        print(f"Loading {model_name} on CPU (this may take a moment)...")
        import torch
        torch.set_num_threads(2)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = 256
        print("[OK] Model loaded")
    
    def search(self,
               user_skills: List[str],
               education: str,
               city: str,
               max_distance_km: int = 50,
               min_stipend: int = 0,
               top_k: int = 10) -> List[Dict]:
        """
        Hybrid search with semantic + lexical fusion and business rules.
        """
        # 1. Encode user profile (mirrors document embedding format)
        user_vector = self._encode_user_profile(user_skills, education, city)
        
        # 2. Semantic search (FAISS)
        distances, indices = self.index.search(
            user_vector.reshape(1, -1).astype('float32'), 
            top_k * 5  # Extra candidates for filtering
        )
        
        semantic_candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            internship_id = self.id_mapping[idx]
            semantic_score = 1.0 - (dist / 2.0)  # L2 → similarity
            semantic_candidates.append((internship_id, max(0.0, min(1.0, semantic_score))))
        
        # 3. Lexical search (FTS5 with BM25)
        lexical_candidates = self._fts5_search(" ".join(user_skills), top_k * 5)
        
        # 4. Fuse with pure Reciprocal Rank Fusion (RRF)
        fused = self._fuse_results(semantic_candidates, lexical_candidates)
        
        # 5. Apply filters + business rules + skill overlap scoring
        filtered = self._apply_filters_and_scoring(
            fused, user_skills, education, min_stipend, city, max_distance_km, top_k
        )
        
        return filtered
    
    def _encode_user_profile(self, skills: List[str], education: str, city: str) -> np.ndarray:
        """
        Encode user profile using the SAME text structure as document embeddings.
        
        Document embeddings use:
            Role: {profile}
            Skills: {skills}
            Company: {company}
            Location: {location}
            Duration: {duration} months
            Education: {education}
            Perks: {perks}
        
        Query mirrors this structure with available user fields.
        """
        skills_text = ", ".join(skills)
        
        text = f"""Role: internship
Skills: {skills_text}
Location: {city}
Education: {education}"""
        
        return self.model.encode([text], normalize_embeddings=True)[0]
    
    def _fts5_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Lexical search using SQLite FTS5 with BM25"""
        try:
            cursor = self.conn.execute("""
                SELECT id, bm25(fts_internships) as rank 
                FROM fts_internships
                WHERE fts_internships MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k))
            
            results = []
            for row in cursor.fetchall():
                score = 1.0 / (1.0 + abs(row[1]))
                results.append((row[0], min(1.0, score * 2.0)))
            
            return results
        except Exception as e:
            logger.warning(f"FTS5 search failed: {e}, falling back to keyword search")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback keyword search"""
        keywords = query.lower().split()
        
        cursor = self.conn.execute("""
            SELECT id, profile, skills FROM internships
        """)
        
        results = []
        for row in cursor.fetchall():
            internship_id, profile, skills_json = row
            text = f"{profile} {skills_json}".lower()
            matches = sum(1 for kw in keywords if kw in text)
            
            if matches > 0:
                score = matches / len(keywords)
                results.append((internship_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _fuse_results(self, 
                     semantic: List[Tuple[str, float]], 
                     lexical: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Pure Reciprocal Rank Fusion (RRF).
        60% semantic weight + 40% lexical weight.
        Only uses rank positions — no raw score mixing.
        """
        fused = {}
        k = 60  # RRF constant
        
        # Semantic: 60% weight
        for rank, (internship_id, _score) in enumerate(semantic):
            rrf_score = 1.0 / (k + rank + 1)
            fused[internship_id] = fused.get(internship_id, 0) + 0.6 * rrf_score
        
        # Lexical: 40% weight
        for rank, (internship_id, _score) in enumerate(lexical):
            rrf_score = 1.0 / (k + rank + 1)
            fused[internship_id] = fused.get(internship_id, 0) + 0.4 * rrf_score
        
        return fused
    
    def _apply_filters_and_scoring(self,
                                  fused_scores: Dict[str, float],
                                  user_skills: List[str],
                                  education: str,
                                  min_stipend: int,
                                  user_city: str,
                                  max_distance_km: int,
                                  top_k: int) -> List[Dict]:
        """Apply hard filters + skill overlap bonus + distance scoring"""
        # Get top candidates by fused score
        candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        
        if not candidates:
            return []
        
        # Find max fused score for normalization (no more arbitrary *2.0 clamping)
        max_fused = max(score for _, score in candidates) if candidates else 1.0
        
        # Fetch metadata
        placeholders = ','.join('?' for _ in candidates)
        cursor = self.conn.execute(f"""
            SELECT 
                id, profile, company, location_normalized,
                stipend_min, stipend_max, duration_months,
                education_req, skills, perks, apply_by, freshness_score,
                role_type, seniority
            FROM internships
            WHERE id IN ({placeholders})
        """, [c[0] for c in candidates])
        
        metadata_map = {}
        for row in cursor.fetchall():
            metadata_map[row[0]] = {
                "id": row[0], "profile": row[1], "company": row[2], "city": row[3],
                "stipend_min": row[4], "stipend_max": row[5], "duration_months": row[6],
                "education_req": row[7], "skills": row[8], "perks": row[9],
                "apply_by": row[10], "freshness_score": row[11],
                "role_type": row[12] if len(row) > 12 else "general",
                "seniority": row[13] if len(row) > 13 else "entry-level / student"
            }
        
        # Prepare user skills set for overlap scoring
        user_skills_lower = set(s.lower().strip() for s in user_skills)
        
        # Apply filters + final scoring
        results = []
        for internship_id, hybrid_score in candidates:
            meta = metadata_map.get(internship_id)
            if not meta:
                continue
            
            # --- HARD FILTERS ---
            
            # Education: hierarchy-based (B.Tech user can see Diploma roles)
            if not is_education_eligible(education, meta["education_req"]):
                continue
            
            # Stipend: check MAX stipend against user's minimum requirement
            if meta["stipend_max"] < min_stipend:
                continue
            
            # Location: distance-based filter
            distance_km = get_city_distance(user_city, meta["city"])
            if distance_km > max_distance_km:
                continue
            
            # --- SCORING ---
            
            # Base score: normalized hybrid score (0-1)
            base_score = hybrid_score / max_fused if max_fused > 0 else 0
            
            # Distance factor (0.3-1.0)
            distance_factor = max(0.3, 1.0 - (distance_km / max_distance_km)) if max_distance_km > 0 else 1.0
            
            # Seniority penalty (demote senior roles)
            seniority = meta.get("seniority", "entry-level / student")
            seniority_factor = 0.6 if "senior" in seniority.lower() else 1.0
            
            # Skill overlap bonus (direct skill matching — new!)
            try:
                job_skills = json.loads(meta["skills"]) if meta["skills"] else []
            except (json.JSONDecodeError, TypeError):
                job_skills = meta["skills"].split(",") if meta["skills"] else []
            
            job_skills_lower = set(s.lower().strip() for s in job_skills)
            
            # Exact match count
            exact_overlap = len(user_skills_lower & job_skills_lower)
            # Partial match (e.g., "python" in "python3")
            partial_overlap = sum(
                1 for us in user_skills_lower 
                for js in job_skills_lower 
                if (us in js or js in us) and us not in job_skills_lower
            )
            
            total_overlap = exact_overlap + partial_overlap * 0.5
            skill_bonus = 1.0 + (total_overlap / max(len(user_skills_lower), 1)) * 0.4
            
            # Final score (0-100)
            final_score = (
                base_score *        # Semantic+lexical relevance (0-1)
                distance_factor *   # Location proximity (0.3-1.0)
                seniority_factor *  # Skill depth match (0.6-1.0)
                skill_bonus *       # Direct skill overlap (1.0-1.4)
                100                 # Scale to 0-100
            )
            
            results.append({
                "id": meta["id"],
                "role": meta["profile"],
                "company": meta["company"],
                "city": meta["city"],
                "stipend_min": meta["stipend_min"],
                "stipend_max": meta["stipend_max"],
                "duration_months": meta["duration_months"],
                "education_req": meta["education_req"],
                "skills": job_skills,
                "perks": meta["perks"],
                "apply_by": meta["apply_by"],
                "match_score": round(min(100.0, final_score), 1),
                "distance_km": round(distance_km, 1),
                "freshness_score": meta["freshness_score"]
            })
        
        # Sort by final score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]
    
    def get_health(self) -> Dict:
        """Health check"""
        total = self.conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
        return {
            "status": "healthy",
            "database_connected": True,
            "model_loaded": self.model is not None,
            "faiss_index_loaded": self.index is not None,
            "total_internships": total,
            "vector_count": self.index.ntotal,
            "search_type": "hybrid (FAISS + FTS5 BM25)",
            "version": "2.1.0"
        }
    
    def close(self):
        if self.conn:
            self.conn.close()

# Singleton instance
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine
