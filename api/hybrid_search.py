"""
Production-ready hybrid search: FAISS (semantic) + SQLite (lexical) + filters
"""
import sqlite3
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, DATA_DIR

# Check if faiss is installed
try:
    import faiss
except ImportError:
    print("❌ FAISS not installed! Install with: pip install faiss-cpu")
    sys.exit(1)

from sentence_transformers import SentenceTransformer
from api.utils import get_city_distance

class HybridSearchEngine:
    """Production-ready hybrid search: FAISS (semantic) + SQLite (lexical) + filters"""
    
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
        # Reduce memory usage
        torch.set_num_threads(2)  # Limit CPU threads
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = 256  # Reduce sequence length to save memory
        print("[OK] Model loaded")
    
    def search(self,
               user_skills: List[str],
               education: str,
               city: str,
               max_distance_km: int = 50,
               min_stipend: int = 0,
               top_k: int = 10) -> List[Dict]:
        """
        Industry-grade hybrid search with business rules
        """
        # 1. Encode user profile WITH skill depth signals
        user_vector = self._encode_user_profile(user_skills, city)
        
        # 2. Semantic search (FAISS)
        distances, indices = self.index.search(
            user_vector.reshape(1, -1).astype('float32'), 
            top_k * 5  # Get extra candidates for filtering
        )
        
        semantic_candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            internship_id = self.id_mapping[idx]
            semantic_score = 1.0 - (dist / 2.0)  # L2 → similarity
            semantic_candidates.append((internship_id, max(0.0, min(1.0, semantic_score))))
        
        # 3. Lexical search (FTS5 with BM25)
        lexical_candidates = self._fts5_search(" ".join(user_skills), top_k * 5)
        
        # 4. Fuse with Reciprocal Rank Fusion (RRF)
        fused = self._fuse_results(semantic_candidates, lexical_candidates)
        
        # 5. Apply hard filters + business rules
        filtered = self._apply_filters_and_scoring(
            fused, education, min_stipend, city, max_distance_km, top_k
        )
        
        return filtered
    
    def _encode_user_profile(self, skills: List[str], city: str) -> np.ndarray:
        """Encode with skill depth awareness"""
        skill_level = "beginner" if len(skills) <= 3 else "intermediate"
        skills_text = ", ".join(skills)
        
        text = f"""Skill Level: {skill_level}
Skills: {skills_text}
Location: {city}
Seeking: entry-level internship for students"""
        
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
        except:
            # Fallback to keyword search if FTS5 not available
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback keyword search"""
        keywords = query.lower().split()
        
        # Search in profile and skills fields
        cursor = self.conn.execute("""
            SELECT id, profile, skills FROM internships
        """)
        
        results = []
        for row in cursor.fetchall():
            internship_id, profile, skills_json = row
            
            # Calculate keyword match score
            text = f"{profile} {skills_json}".lower()
            matches = sum(1 for kw in keywords if kw in text)
            
            if matches > 0:
                score = matches / len(keywords)  # Normalize by query length
                results.append((internship_id, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _fuse_results(self, 
                     semantic: List[Tuple[str, float]], 
                     lexical: List[Tuple[str, float]]) -> Dict[str, float]:
        """Reciprocal Rank Fusion: 0.7 semantic + 0.3 lexical"""
        fused = {}
        
        # Semantic contribution (70%)
        for rank, (internship_id, score) in enumerate(semantic):
            fused[internship_id] = fused.get(internship_id, 0) + (0.7 * score)
        
        # Lexical contribution (30%)
        for rank, (internship_id, score) in enumerate(lexical):
            fused[internship_id] = fused.get(internship_id, 0) + (0.3 * score)
        
        return fused
    
    def _apply_filters_and_scoring(self,
                                  fused_scores: Dict[str, float],
                                  education: str,
                                  min_stipend: int,
                                  user_city: str,
                                  max_distance_km: int,
                                  top_k: int) -> List[Dict]:
        """Apply hard filters + business rules (freshness, distance)"""
        # Get top candidates by fused score
        candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        
        if not candidates:
            return []
        
        # Fetch metadata for filtering
        placeholders = ','.join('?' for _ in candidates)
        cursor = self.conn.execute(f"""
            SELECT 
                id, profile, company, location_normalized,
                stipend_min, stipend_max, duration_months,
                education_req, skills, perks, apply_by, freshness_score
            FROM internships
            WHERE id IN ({placeholders})
        """, [c[0] for c in candidates])
        
        metadata_map = {}
        for row in cursor.fetchall():
            metadata_map[row[0]] = {
                "id": row[0], "profile": row[1], "company": row[2], "city": row[3],
                "stipend_min": row[4], "stipend_max": row[5], "duration_months": row[6],
                "education_req": row[7], "skills": row[8], "perks": row[9],
                "apply_by": row[10], "freshness_score": row[11]
            }
        
        # Apply filters + final scoring
        results = []
        for internship_id, hybrid_score in candidates:
            meta = metadata_map.get(internship_id)
            if not meta:
                continue
            
            # Hard filters
            if meta["education_req"] != "Any" and meta["education_req"] != education:
                continue
            if meta["stipend_min"] < min_stipend:
                continue
            
            # Location filter using distance matrix
            distance_km = get_city_distance(user_city, meta["city"])
            if distance_km > max_distance_km:
                continue
            
            # Final score = hybrid × freshness × distance factor
            distance_factor = max(0.5, 1.0 - (distance_km / max_distance_km)) if max_distance_km > 0 else 1.0
            final_score = hybrid_score * meta["freshness_score"] * distance_factor * 100
            
            # Parse skills from JSON string
            try:
                skills = json.loads(meta["skills"]) if meta["skills"] else []
            except:
                skills = meta["skills"].split(",") if meta["skills"] else []
            
            results.append({
                "id": meta["id"],
                "role": meta["profile"],
                "company": meta["company"],
                "city": meta["city"],
                "stipend_min": meta["stipend_min"],
                "stipend_max": meta["stipend_max"],
                "duration_months": meta["duration_months"],
                "education_req": meta["education_req"],
                "skills": skills,
                "perks": meta["perks"],
                "apply_by": meta["apply_by"],
                "match_score": round(final_score, 1),
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
            "search_type": "hybrid (FAISS + FTS5 BM25)"
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
