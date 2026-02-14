import json
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CITY_DISTANCE_MATRIX

_DISTANCE_MATRIX = None

def load_distance_matrix() -> Dict[str, Dict[str, float]]:
    global _DISTANCE_MATRIX
    if _DISTANCE_MATRIX is None:
        if CITY_DISTANCE_MATRIX.exists():
            with open(CITY_DISTANCE_MATRIX, 'r') as f:
                _DISTANCE_MATRIX = json.load(f)
        else:
            _DISTANCE_MATRIX = {}
    return _DISTANCE_MATRIX

def get_city_distance(city1: str, city2: str) -> float:
    matrix = load_distance_matrix()
    if city1 == "Remote" or city2 == "Remote":
        return 0.0
    if city1 in matrix and city2 in matrix[city1]:
        return matrix[city1][city2]
    return 9999.0

def calculate_final_score(vec_distance: float, freshness_score: float, 
                         distance_km: float, max_distance: float) -> float:
    semantic_score = (1.0 - vec_distance) * 50.0
    freshness_boost = freshness_score * 30.0
    distance_score = (1.0 - distance_km / max_distance) * 20.0 if distance_km <= max_distance else 0.0
    return min(100.0, max(0.0, semantic_score + freshness_boost + distance_score))
