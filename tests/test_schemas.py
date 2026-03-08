import pytest
from pydantic import ValidationError
from api.schemas import UserProfile, InternshipResponse

def test_user_profile_validation():
    # Valid profile
    profile = UserProfile(
        skills=["Python", "React"],
        education="B.Tech",
        city="Mumbai",
        max_distance_km=50,
        min_stipend=10000
    )
    assert profile.skills == ["Python", "React"]
    assert profile.max_distance_km == 50
    
    # Invalid skills (empty list)
    with pytest.raises(ValidationError):
        UserProfile(
            skills=[],
            education="B.Tech",
            city="Mumbai"
        )
    
    # Invalid skills (empty string skill)
    with pytest.raises(ValidationError):
        UserProfile(
            skills=["", " "],
            education="B.Tech",
            city="Mumbai"
        )
    
    # Invalid max_distance_km (out of range)
    with pytest.raises(ValidationError):
        UserProfile(
            skills=["Python"],
            education="B.Tech",
            city="Mumbai",
            max_distance_km=600  # Max is 500
        )
    
    # Negative min_stipend
    with pytest.raises(ValidationError):
        UserProfile(
            skills=["Python"],
            education="B.Tech",
            city="Mumbai",
            min_stipend=-100
        )

def test_internship_response_validation():
    # Valid internship response
    internship = InternshipResponse(
        id="123",
        role="Backend Intern",
        company="Tech Corp",
        location="Mumbai",
        city="Mumbai",
        stipend_min=10000,
        stipend_max=15000,
        duration_months=6,
        education_req="B.Tech",
        skills=["Python", "SQL"],
        perks="None",
        apply_by="2 Oct' 25",
        match_score=85.5,
        distance_km=5.0,
        freshness_score=0.9
    )
    assert internship.id == "123"
    assert internship.match_score == 85.5
    
    # Invalid match score (out of range 0-100)
    with pytest.raises(ValidationError):
        InternshipResponse(
            id="123",
            role="Backend Intern",
            company="Tech Corp",
            location="Mumbai",
            city="Mumbai",
            stipend_min=10000,
            stipend_max=15000,
            duration_months=6,
            education_req="B.Tech",
            skills=["Python", "SQL"],
            match_score=105.0, # Too high
            distance_km=5.0,
            freshness_score=0.9
        )
