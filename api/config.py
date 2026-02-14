from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    DATABASE_PATH: str = "database/internships.db"
    API_TITLE: str = "Internship Recommender API"
    API_VERSION: str = "2.0.0"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    DEFAULT_TOP_K: int = 10
    MAX_DISTANCE_KM: int = 100
    MIN_STIPEND_DEFAULT: int = 0
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
