import sqlite3
import json
import numpy as np
from typing import List, Dict, Any
from api.config import settings
from api.utils import get_city_distance

class Database:
    def __init__(self):
        self.db_path = settings.DATABASE_PATH
        self.conn = None
        self._init_connection()
    
    def _init_connection(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
    
    def get_internship_by_id(self, internship_id: str) -> Dict[str, Any]:
        cursor = self.conn.execute("SELECT * FROM internships WHERE id = ?", (internship_id,))
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row))
        if result.get('skills'):
            result['skills'] = json.loads(result['skills'])
        return result
    
    def search_internships(self, user_vector: np.ndarray, education: str, 
                          min_stipend: int, user_city: str, 
                          max_distance: int, top_k: int = 50) -> List[Dict[str, Any]]:
        # Get all candidates matching filters
        results = self.conn.execute("""
            SELECT * FROM internships
            WHERE education_req IN (?, 'Any')
              AND stipend_min >= ?
            ORDER BY freshness_score DESC
            LIMIT ?
        """, (education, min_stipend, top_k * 3)).fetchall()
        
        columns = [desc[0] for desc in self.conn.execute("SELECT * FROM internships LIMIT 1").description]
        
        filtered_results = []
        for row in results:
            internship = dict(zip(columns, row))
            
            # Calculate distance
            city = internship['location_normalized']
            distance_km = get_city_distance(user_city, city)
            
            if distance_km > max_distance:
                continue
            
            # Calculate vector similarity
            emb_bytes = internship['embedding']
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            vec_distance = float(np.linalg.norm(user_vector - emb))
            
            internship['vec_distance'] = vec_distance
            internship['distance_km'] = distance_km
            if internship.get('skills'):
                internship['skills'] = json.loads(internship['skills'])
            
            filtered_results.append(internship)
        
        # Sort by vector distance
        filtered_results.sort(key=lambda x: x['vec_distance'])
        return filtered_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        stats['total_internships'] = self.conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0]
        stats['internships_by_city'] = dict(self.conn.execute("""
            SELECT location_normalized, COUNT(*) FROM internships GROUP BY location_normalized
        """).fetchall())
        avg_stipend = self.conn.execute("""
            SELECT AVG(stipend_min), AVG(stipend_max) FROM internships WHERE stipend_min > 0
        """).fetchone()
        stats['avg_stipend'] = {
            'min': round(avg_stipend[0], 0) if avg_stipend[0] else 0,
            'max': round(avg_stipend[1], 0) if avg_stipend[1] else 0
        }
        return stats
    
    def close(self):
        if self.conn:
            self.conn.close()
