from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.config import settings
from api.schemas import UserProfile, RecommendationResponse, InternshipResponse, HealthCheck
from api.engine_selector import get_search_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info("Starting Internship Recommender API v2.0")
    engine = get_search_engine()
    logger.info("Ready to serve recommendations!")

@app.on_event("shutdown")
async def shutdown_event():
    if engine:
        engine.close()
        logger.info("Connections closed")

@app.get("/", response_model=HealthCheck)
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Service starting up")
    health = engine.get_health()
    return HealthCheck(**health)

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(profile: UserProfile):
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Request: skills={profile.skills}, city={profile.city}")
        results = engine.search(
            user_skills=profile.skills,
            education=profile.education,
            city=profile.city,
            max_distance_km=profile.max_distance_km,
            min_stipend=profile.min_stipend,
            top_k=settings.DEFAULT_TOP_K
        )
        
        recommendations = [
            InternshipResponse(
                id=r['id'],
                role=r['role'],
                company=r['company'],
                location=r['city'],
                city=r['city'],
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
        
        logger.info(f"Returned {len(recommendations)} recommendations")
        return RecommendationResponse(
            query=profile,
            total_results=len(recommendations),
            recommendations=recommendations,
            metadata={"version": settings.API_VERSION, "model": settings.EMBEDDING_MODEL}
        )
    
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recommendation service error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
