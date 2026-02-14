import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from api.config import settings
from api.database import Database
from api.utils import calculate_final_score

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.db = None
        self._load_model()
        self._init_db()
    
    def _load_model(self):
        print(f"🔄 Loading model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
        print(f"✅ Model loaded on {settings.EMBEDDING_DEVICE}")
    
    def _init_db(self):
        self.db = Database()
        print(f"✅ Database connected")
    
    def recommend(self, skills: List[str], education: str, city: str,
                 max_distance_km: int, min_stipend: int, top_k: int = 10) -> List[Dict[str, Any]]:
        # Create query text
        query_text = f"Skills: {', '.join(skills)}\nEducation: {education}\nLocation: {city}"
        
        # Generate embedding
        query_embedding = self.model.encode(query_text, normalize_embeddings=True, convert_to_numpy=True)
        
        # Search database
        candidates = self.db.search_internships(
            user_vector=query_embedding,
            education=education,
            min_stipend=min_stipend,
            user_city=city,
            max_distance=max_distance_km,
            top_k=top_k * 2
        )
        
        # Calculate final scores
        for internship in candidates:
            internship['match_score'] = calculate_final_score(
                vec_distance=internship['vec_distance'],
                freshness_score=internship['freshness_score'],
                distance_km=internship['distance_km'],
                max_distance=max_distance_km
            )
        
        # Sort by match score
        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        
        return candidates[:top_k]
    
    def get_health(self) -> Dict[str, Any]:
        stats = self.db.get_stats()
        return {
            "status": "healthy",
            "database_connected": True,
            "model_loaded": self.model is not None,
            "total_internships": stats['total_internships'],
            "version": settings.API_VERSION
        }
