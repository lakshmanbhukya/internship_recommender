import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

def get_mongo_collection():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]
    return collection
