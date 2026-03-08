import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import api.main
from api.main import app

client = TestClient(app)

@pytest.fixture
def mock_engine(monkeypatch):
    """Fixture to mock the search engine"""
    mock = MagicMock()
    mock.get_health.return_value = {
        "status": "healthy",
        "database_connected": True,
        "model_loaded": True,
        "total_internships": 100,
        "version": "2.0.0"
    }
    
    mock.search.return_value = [
        {
            "id": "test_1",
            "role": "Software Engineer Intern",
            "company": "Test Company",
            "city": "Mumbai",
            "stipend_min": 10000,
            "stipend_max": 20000,
            "duration_months": 6,
            "education_req": "B.Tech",
            "skills": ["Python", "React"],
            "match_score": 95.0,
            "distance_km": 5.0,
            "freshness_score": 0.9,
            "perks": "Flexible Hours",
            "apply_by": "2 Oct' 25"
        }
    ]
    
    # Force the app engine to be our mock
    monkeypatch.setattr(api.main, "engine", mock)
    return mock

def test_health_check(mock_engine):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_internships"] == 100

def test_recommend_endpoint(mock_engine):
    profile_data = {
        "skills": ["Python", "React"],
        "education": "B.Tech",
        "city": "Mumbai",
        "max_distance_km": 50,
        "min_stipend": 5000
    }
    response = client.post("/recommend", json=profile_data)
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert data["recommendations"][0]["role"] == "Software Engineer Intern"
    
    # Verify the mock search was called with correct arguments
    mock_engine.search.assert_called_once()
    args, kwargs = mock_engine.search.call_args
    assert kwargs["city"] == "Mumbai"
    assert "Python" in kwargs["user_skills"]

def test_recommend_endpoint_invalid_input(mock_engine):
    # Missing education
    profile_data = {
        "skills": ["Python"],
        "city": "Mumbai"
    }
    response = client.post("/recommend", json=profile_data)
    assert response.status_code == 422 # Pydantic validation error

def test_recommend_endpoint_service_not_ready(monkeypatch):
    # Test case when engine is None
    monkeypatch.setattr(api.main, "engine", None)
    response = client.get("/")
    assert response.status_code == 503
    assert response.json()["detail"] == "Service starting up"

    response = client.post("/recommend", json={
        "skills": ["Python"],
        "education": "B.Tech",
        "city": "Mumbai"
    })
    assert response.status_code == 503
    assert response.json()["detail"] == "Service not ready"
