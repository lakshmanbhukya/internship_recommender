from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.config import settings
from api.schemas import UserProfile, RecommendationResponse, InternshipResponse, HealthCheck
from api.recommendations import RecommendationEngine

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("🚀 Starting Internship Recommender API v2.0")
    engine = RecommendationEngine()
    print("✅ Ready to serve recommendations!")

@app.get("/", response_model=HealthCheck)
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Service starting up")
    return engine.get_health()

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(profile: UserProfile):
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        results = engine.recommend(
            skills=profile.skills,
            education=profile.education,
            city=profile.city,
            max_distance_km=profile.max_distance_km,
            min_stipend=profile.min_stipend,
            top_k=settings.DEFAULT_TOP_K
        )
        
        recommendations = [
            InternshipResponse(
                id=r['id'],
                role=r['profile'],
                company=r['company'],
                location=r['location_original'],
                city=r['location_normalized'],
                stipend_min=r['stipend_min'],
                stipend_max=r['stipend_max'],
                duration_months=r['duration_months'],
                education_req=r['education_req'],
                skills=r['skills'],
                perks=r.get('perks'),
                apply_by=r.get('apply_by'),
                match_score=r['match_score'],
                distance_km=r['distance_km'],
                freshness_score=r['freshness_score']
            )
            for r in results
        ]
        
        return RecommendationResponse(
            query=profile,
            total_results=len(recommendations),
            recommendations=recommendations,
            metadata={"version": settings.API_VERSION, "model": settings.EMBEDDING_MODEL}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
