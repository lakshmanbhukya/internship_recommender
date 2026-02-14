import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed" / "internships_cleaned.csv"
GEOCODING_CACHE = DATA_DIR / "geocoding_cache.json"
CITY_DISTANCE_MATRIX = DATA_DIR / "city_distance_matrix.json"
DB_PATH = BASE_DIR / "database" / "internships.db"

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# City normalization
CITY_MAPPINGS = {
    "Bangalore": "Bangalore", "Bengaluru": "Bangalore",
    "Mumbai": "Mumbai", "New Delhi": "Delhi", "Delhi": "Delhi",
    "Noida": "Noida", "Gurgaon": "Gurgaon", "Gurugram": "Gurgaon",
    "Chennai": "Chennai", "Hyderabad": "Hyderabad", "Pune": "Pune",
    "Kolkata": "Kolkata", "Ahmedabad": "Ahmedabad",
    "Work from home": "Remote", "Remote": "Remote", "Anywhere in India": "Remote"
}

# Education normalization
EDU_MAPPINGS = {
    "Not Specified": "Any", "B.Tech": "B.Tech", "B.E.": "B.Tech",
    "M.Tech": "M.Tech", "MBA": "MBA", "B.Com": "B.Com", "M.Com": "M.Com",
    "B.Sc": "B.Sc", "M.Sc": "M.Sc", "B.A": "B.A", "M.A": "M.A",
    "Diploma": "Diploma", "PhD": "PhD"
}

DEFAULT_STIPEND = {"min": 0, "max": 0}
DEFAULT_DURATION = 3
