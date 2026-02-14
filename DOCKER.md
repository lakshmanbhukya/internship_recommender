# Docker Commands

## Build
```bash
docker build -t internship-recommender .
```

## Run Locally
```bash
docker run -p 8000:8000 internship-recommender
```

## Tag for Docker Hub
```bash
docker tag internship-recommender your-username/internship-recommender:latest
```

## Push to Docker Hub
```bash
docker login
docker push your-username/internship-recommender:latest
```

## Pull and Run from Docker Hub
```bash
docker pull your-username/internship-recommender:latest
docker run -p 8000:8000 your-username/internship-recommender:latest
```

## GitHub Actions (Automated)
Push to `main` or `refactor/v2` branch triggers:
1. Run tests
2. Build Docker image
3. Push to Docker Hub as `latest` and `<commit-sha>`

## Image Size
- Base: ~150 MB
- With dependencies: ~2.5 GB (includes PyTorch, transformers)
- Database + FAISS: ~75 MB
