import pytest
from api.utils import is_education_eligible, get_city_distance, parse_apply_by_date
from datetime import datetime

def test_is_education_eligible():
    # EDUCATION_HIERARCHY: PhD: 6, M.Tech: 5, B.Tech: 4, Diploma: 3, Any: 1
    assert is_education_eligible("B.Tech", "B.Tech") == True
    assert is_education_eligible("M.Tech", "B.Tech") == True
    assert is_education_eligible("B.Tech", "M.Tech") == False
    assert is_education_eligible("Any", "Any") == True
    assert is_education_eligible("B.Tech", "Any") == True
    assert is_education_eligible("B.Tech", "Diploma") == True
    assert is_education_eligible("Diploma", "B.Tech") == False
    assert is_education_eligible("Random", "Any") == True

def test_get_city_distance():
    # Remote cases
    assert get_city_distance("Remote", "Mumbai") == 0.0
    assert get_city_distance("Delhi", "Remote") == 0.0
    
    # Same city
    assert get_city_distance("Mumbai", "Mumbai") == 0.0
    
    # Suburban mappings
    # Faridabad -> Delhi (mapping in utils.py)
    # If both are resolved to Delhi, distance is 0.0
    assert get_city_distance("Faridabad", "Delhi") == 0.0
    
    # Unknown cities
    assert get_city_distance("UnknownCity1", "UnknownCity2") == 300.0

def test_parse_apply_by_date():
    # Valid date formats
    d1 = parse_apply_by_date("2 Oct' 25")
    assert isinstance(d1, datetime)
    assert d1.day == 2
    assert d1.month == 10
    assert d1.year == 2025
    
    d2 = parse_apply_by_date("15 Nov' 25")
    assert isinstance(d2, datetime)
    assert d2.day == 15
    assert d2.month == 11
    assert d2.year == 2025

    d3 = parse_apply_by_date("2025-10-02")
    assert isinstance(d3, datetime)
    assert d3.day == 2
    assert d3.month == 10
    assert d3.year == 2025

    # Invalid date
    assert parse_apply_by_date("invalid date") is None
    assert parse_apply_by_date("") is None
