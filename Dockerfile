# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-new.txt .
RUN pip install --no-cache-dir --user -r requirements-new.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application files
COPY config/ config/
COPY api/ api/
COPY database/internships.db database/internships.db
COPY data/city_distance_matrix.json data/city_distance_matrix.json
COPY data/geocoding_cache.json data/geocoding_cache.json
COPY data/faiss_index.bin data/faiss_index.bin
COPY data/id_mapping.json data/id_mapping.json

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
