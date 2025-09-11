import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from sklearn.metrics.pairwise import cosine_similarity
from connection import get_mongo_collection  # Your Mongo connection helper

try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    raise FileNotFoundError("TF-IDF vectorizer file not found. Please ensure 'tfidf_vectorizer.joblib' exists.")


def combine_text_features(df):
    """Extract text combination logic to eliminate code duplication"""
    return df['title'] + " " + df['skills'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def recommend_internship_mongodb(skills_str, sectors_str, education_level, city_name, max_distance_km=150):
    try:
        collection = get_mongo_collection()
    except Exception as e:
        return {"error": f"Database connection failed: {str(e)}", "nearby_ids": [], "remote_ids": []}
    
    sector_list = [s.strip().lower() for s in sectors_str.split(',')]
    geolocator = Nominatim(user_agent="internship_recommender")
    location = geolocator.geocode(city_name)

    if location is None:
        return {"error": "Could not determine location for the city entered.", "nearby_ids": [], "remote_ids": []}

    candidate_lat, candidate_lon = round(location.latitude, 4), round(location.longitude, 4)
    candidate_skills = skills_str.lower()
    candidate_education = education_level.lower()

    pipeline = [
        {
            "$match": {
                "sector": {"$in": sector_list},
                "min_education": {"$lte": candidate_education}
            }
        },
        {
            "$addFields": {
                "distance_km": {
                    "$multiply": [
                        111,
                        {
                            "$sqrt": {
                                "$add": [
                                    {"$pow": [{"$subtract": ["$latitude", candidate_lat]}, 2]},
                                    {"$pow": [{"$subtract": ["$longitude", candidate_lon]}, 2]}
                                ]
                            }
                        }
                    ]
                }
            }
        },
        {
            "$match": {
                "distance_km": {"$lte": max_distance_km}
            }
        },
        {
            "$limit": 500
        }
    ]

    try:
        filtered_data = list(collection.aggregate(pipeline))
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}", "nearby_ids": [], "remote_ids": []}

    nearby_ids = []
    remote_ids = []

    if filtered_data:
        df = pd.DataFrame(filtered_data)

        # Precise distance calculation
        df['precise_distance_km'] = df.apply(
            lambda row: haversine_dist(candidate_lat, candidate_lon, row['latitude'], row['longitude']), axis=1)
        df = df[df['precise_distance_km'] <= max_distance_km]

        if not df.empty:
            df['combined_text'] = combine_text_features(df)
            internship_tfidf = vectorizer.transform(df['combined_text'])
            candidate_tfidf = vectorizer.transform([candidate_skills])
            similarity_scores = cosine_similarity(candidate_tfidf, internship_tfidf).flatten()
            df['similarity_score'] = similarity_scores
            df['final_score'] = df.apply(
                lambda row: row['similarity_score'] + max(0, (max_distance_km - row['precise_distance_km']) / max_distance_km),
                axis=1
            )
            df = df.sort_values(by='final_score', ascending=False)
            top_nearby_n = min(5, len(df))
            nearby_ids = df.head(top_nearby_n)['_id'].apply(str).tolist()

    if len(nearby_ids) < 5:
        # Need to supplement with remote internships that match profile
        remote_filter = {
            "sector": {"$in": sector_list},
            "min_education": {"$lte": candidate_education},
            "mode": "remote"
        }
        try:
            remote_data = list(collection.find(remote_filter).limit(10))
        except Exception as e:
            return {"error": f"Remote query failed: {str(e)}", "nearby_ids": nearby_ids, "remote_ids": []}

        if remote_data:
            df_remote = pd.DataFrame(remote_data)
            df_remote['combined_text'] = combine_text_features(df_remote)
            internship_tfidf_remote = vectorizer.transform(df_remote['combined_text'])
            candidate_tfidf_remote = vectorizer.transform([candidate_skills])
            similarity_scores_remote = cosine_similarity(candidate_tfidf_remote, internship_tfidf_remote).flatten()
            df_remote['similarity_score'] = similarity_scores_remote
            df_remote['final_score'] = df_remote['similarity_score']

            df_remote = df_remote.sort_values(by='final_score', ascending=False)
            top_remote_n = min(5, len(df_remote))
            remote_ids = df_remote.head(top_remote_n)['_id'].apply(str).tolist()

    return {
        "nearby_ids": nearby_ids,
        "remote_ids": remote_ids
    }
