FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-new.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY config/ config/
COPY api/ api/
COPY database/ database/
COPY data/city_distance_matrix.json data/
COPY data/geocoding_cache.json data/

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
