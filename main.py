from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from recommender import recommend_internship_mongodb

app = FastAPI()

class RecommendRequest(BaseModel):
    skills: str
    sectors: str
    education_level: str
    city_name: str
    max_distance_km: Optional[int] = Field(150, description="Maximum distance in km for internship search")

@app.post("/recommend")
def recommend(data: RecommendRequest):
    results = recommend_internship_mongodb(
        data.skills,
        data.sectors,
        data.education_level,
        data.city_name,
        max_distance_km=data.max_distance_km
    )
    if isinstance(results, str):
        raise HTTPException(status_code=404, detail=results)
    return {"recommendations": results}
