# Internship Recommender System

## Overview
This project is an intelligent internship recommendation system that matches candidates with suitable internship opportunities based on their skills, education level, preferred sectors, and location. The system uses natural language processing and machine learning techniques to provide personalized recommendations.

## Features
- **Semantic Matching**: Uses dense embeddings (nomic-embed-text) for contextual understanding beyond keyword matching
- **Hybrid Search**: Combines TF-IDF (sparse) and embedding (dense) similarities for optimal results
- **Location-Aware Recommendations**: Provides nearby internships using geographic proximity with Haversine formula
- **Education Level Filtering**: Ensures recommendations meet candidate's education qualifications
- **Sector-Specific Matching**: Filters internships by preferred industry sectors
- **Explainable Results**: Provides detailed reasoning for each recommendation
- **Resume Processing**: Supports PDF, DOCX, and text resume parsing

## Technologies Used

### Backend Framework
- **FastAPI**: High-performance web framework for building APIs with Python

### Machine Learning & Embeddings
- **sentence-transformers**: For semantic embeddings using nomic-embed-text
- **scikit-learn**: For TF-IDF vectorization and hybrid matching
- **ChromaDB**: Vector database for efficient similarity search
- **pandas & numpy**: For data manipulation and numerical operations

### Document Processing
- **PyPDF2**: For PDF resume parsing
- **python-docx**: For Word document processing
- **pdfplumber**: Enhanced PDF text extraction

### Geospatial Processing
- **geopy**: For geocoding city names to coordinates
- **Haversine Formula**: For accurate distance calculations

### Database & Storage
- **ChromaDB**: Primary vector database for embeddings and metadata
- **MongoDB**: Optional - only needed for legacy TF-IDF mode
- **PyMongo**: Python driver for MongoDB (legacy mode only)

## Technical Architecture

### Semantic Recommendation Algorithm
1. **Resume Processing**: Extract and normalize text from PDF/DOCX/TXT files
2. **Semantic Profile Creation**: Generate contextual embeddings for candidate skills and preferences
3. **Hybrid Search**: Combine TF-IDF (sparse) and embedding (dense) similarity matching
4. **Rule-Based Filtering**: Apply sector, education, and location constraints
5. **Multi-Factor Ranking**: Weighted scoring using semantic similarity, distance, and skill coverage
6. **Explainable Results**: Generate detailed reasoning for each recommendation

### API Endpoints
- **/recommend**: POST endpoint that accepts candidate preferences and returns personalized internship recommendations

## Setup and Installation

### Prerequisites
- Python 3.7+
- MongoDB instance (optional - only for legacy mode)

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   # ChromaDB Configuration (required)
   CHROMA_PERSIST_DIR=./chroma
   CHROMA_COLLECTION_NAME=internship_vectors
   EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
   
   # MongoDB Configuration (optional - legacy mode only)
   MONGO_URI=your_mongodb_connection_string
   DB_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   ```
4. Ensure the `tfidf_vectorizer.joblib` file is present in the root directory

### Quick Setup
```bash
# Install dependencies and setup
python setup_semantic.py

# Test system components
python test_semantic.py

# Migrate existing data to vector database
python migrate_data.py

# Start the application
uvicorn main:app --reload
```

## Usage
Send a POST request to the `/recommend` endpoint:

### Default Semantic Search
```json
{
  "skills": "python, machine learning, data science",
  "sectors": "technology, finance",
  "education_level": "bachelor",
  "city_name": "Bangalore"
}
```

### Legacy TF-IDF Search (requires MongoDB)
```json
{
  "skills": "python, data analysis",
  "sectors": "technology",
  "education_level": "bachelor",
  "city_name": "Bangalore",
  "use_semantic": false
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

## System Architecture
```
User Input → Semantic Profile → ChromaDB Vector Search → 
Hybrid Ranking → Explainable Recommendations
```

## Performance Features
- **Semantic Understanding**: Context-aware matching beyond keywords
- **Scalable Search**: ANN-based vector retrieval (O(log n))
- **Smart Caching**: TTL-based caching for geocoding and embeddings
- **Hybrid Scoring**: Configurable weights for different similarity factors
- **Backward Compatibility**: Seamless fallback to lexical matching