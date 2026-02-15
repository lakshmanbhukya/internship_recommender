import json
import sys
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CITY_DISTANCE_MATRIX

logger = logging.getLogger(__name__)

# =============================================
# Education Hierarchy
# =============================================
# Matches the actual values in the Internshala 2025 dataset:
# Any, B.A, B.Com, B.Sc, B.Tech, Diploma
EDUCATION_HIERARCHY = {
    "PhD": 6,
    "M.Tech": 5, "MBA": 5, "M.Sc": 5, "M.Com": 5, "M.A": 5,
    "B.Tech": 4, "B.Sc": 4, "B.Com": 4, "B.A": 4,
    "Diploma": 3,
    "Any": 1
}

def is_education_eligible(user_edu: str, required_edu: str) -> bool:
    """
    Check if user's education meets or exceeds the internship requirement.
    e.g. B.Tech user is eligible for Diploma-required internships.
    """
    if required_edu == "Any":
        return True
    user_level = EDUCATION_HIERARCHY.get(user_edu, 3)
    req_level = EDUCATION_HIERARCHY.get(required_edu, 3)
    return user_level >= req_level


# =============================================
# City Distance
# =============================================
# Suburban → parent city mappings for cities not in the distance matrix
SUBURBAN_MAPPINGS = {
    "Faridabad": "Delhi",
    "Ghaziabad": "Delhi",
    "Greater Noida": "Noida",
    "Thane": "Mumbai",
    "Navi Mumbai": "Mumbai",
    "Pimpri-Chinchwad": "Pune",
    "Gurugram": "Gurgaon",
}

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

def _resolve_city(city: str) -> str:
    """Map suburban cities to their parent city for distance lookup."""
    return SUBURBAN_MAPPINGS.get(city, city)

def get_city_distance(city1: str, city2: str) -> float:
    """
    Get distance between two Indian cities.
    Returns 0 for Remote, uses distance matrix for known cities,
    300 km default for unknown cities (reasonable for same-country).
    """
    if city1 == "Remote" or city2 == "Remote":
        return 0.0
    
    # Resolve suburban cities
    c1 = _resolve_city(city1)
    c2 = _resolve_city(city2)
    
    if c1 == c2:
        return 0.0
    
    matrix = load_distance_matrix()
    
    # Try both directions
    if c1 in matrix and c2 in matrix[c1]:
        return matrix[c1][c2]
    if c2 in matrix and c1 in matrix[c2]:
        return matrix[c2][c1]
    
    # Unknown city — use 300 km instead of 9999
    return 300.0


# =============================================
# Date Parsing (Internshala format)
# =============================================
def parse_apply_by_date(date_str: str) -> datetime:
    """
    Parse Internshala's apply_by format: "2 Oct' 25", "15 Nov' 25"
    Returns datetime or None if unparseable.
    """
    if not date_str:
        return None
    try:
        cleaned = date_str.strip().replace("'", "")
        return datetime.strptime(cleaned, "%d %b %y")
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except:
            return None
