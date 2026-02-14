from pydantic import BaseModel, Field, validator
from typing import List, Optional

class UserProfile(BaseModel):
    skills: List[str] = Field(..., min_length=1, max_length=20)
    education: str
    city: str
    max_distance_km: int = Field(default=50, ge=0, le=500)
    min_stipend: int = Field(default=0, ge=0)
    preferred_sectors: List[str] = Field(default_factory=list)
    
    @validator('skills')
    def validate_skills(cls, v):
        if not all(len(s.strip()) > 0 for s in v):
            raise ValueError('Skills cannot be empty strings')
        return [s.strip() for s in v]

class InternshipResponse(BaseModel):
    id: str
    role: str
    company: str
    location: str
    city: str
    stipend_min: int
    stipend_max: int
    duration_months: int
    education_req: str
    skills: List[str]
    perks: Optional[str]
    apply_by: Optional[str]
    match_score: float = Field(..., ge=0, le=100)
    distance_km: float
    freshness_score: float

class RecommendationResponse(BaseModel):
    query: UserProfile
    total_results: int
    recommendations: List[InternshipResponse]
    metadata: dict

class HealthCheck(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    total_internships: int
    version: str
