# Internship Recommender System

## Overview
This project is an intelligent internship recommendation system that matches candidates with suitable internship opportunities based on their skills, education level, preferred sectors, and location. The system uses natural language processing and machine learning techniques to provide personalized recommendations.

## Features
- **Skill-Based Matching**: Uses TF-IDF vectorization and cosine similarity to match candidate skills with internship requirements
- **Location-Aware Recommendations**: Provides nearby internships based on geographic proximity using the Haversine formula
- **Education Level Filtering**: Ensures recommendations meet the candidate's education qualifications
- **Sector-Specific Matching**: Filters internships by preferred industry sectors
- **Remote Work Options**: Supplements recommendations with remote internships when local options are limited

## Technologies Used

### Backend Framework
- **FastAPI**: High-performance web framework for building APIs with Python

### Data Processing & Machine Learning
- **scikit-learn**: Used for TF-IDF vectorization and cosine similarity calculations
- **pandas & numpy**: For efficient data manipulation and numerical operations
- **joblib**: For model serialization and persistence

### Geospatial Processing
- **geopy**: For geocoding city names to coordinates
- **Haversine Formula**: Custom implementation for accurate distance calculations between coordinates

### Database
- **MongoDB**: NoSQL database for storing internship data
- **PyMongo**: Python driver for MongoDB integration

### Environment & Configuration
- **python-dotenv**: For managing environment variables and configuration

## Technical Architecture

### Recommendation Algorithm
1. **Data Filtering**: Initial filtering of internships based on sector and education requirements
2. **Geographic Proximity**: Calculation of distances between candidate location and internship locations
3. **Text Similarity**: TF-IDF vectorization of skills and job descriptions to calculate similarity scores
4. **Hybrid Scoring**: Combination of text similarity and distance metrics for final ranking
5. **Remote Fallback**: Supplementation with remote internships when local options are insufficient

### API Endpoints
- **/recommend**: POST endpoint that accepts candidate preferences and returns personalized internship recommendations

## Setup and Installation

### Prerequisites
- Python 3.7+
- MongoDB instance

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   MONGO_URI=your_mongodb_connection_string
   DB_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   ```
4. Ensure the `tfidf_vectorizer.joblib` file is present in the root directory

### Running the Application
```
uvicorn main:app --reload
```

## Usage
Send a POST request to the `/recommend` endpoint with the following JSON structure:

```json
{
  "skills": "python, data analysis, machine learning",
  "sectors": "technology, finance",
  "education_level": "bachelor",
  "city_name": "New York",
  "max_distance_km": 100
}
```

## Response Format
```json
{
  "recommendations": {
    "nearby_ids": ["id1", "id2", "id3"],
    "remote_ids": ["id4", "id5"]
  }
}
```

## Future Enhancements
- User authentication and profile management
- Improved NLP techniques for better skill matching
- Recommendation feedback and learning system
- Integration with job application platforms
- Mobile application interface