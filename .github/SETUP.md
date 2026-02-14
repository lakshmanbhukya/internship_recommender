# GitHub Actions Setup

## Required Secrets

Add these secrets in GitHub repository settings:

1. Go to: `Settings` → `Secrets and variables` → `Actions`
2. Click `New repository secret`
3. Add:

### DOCKER_USERNAME
Your Docker Hub username

### DOCKER_PASSWORD
Your Docker Hub access token (not password)
- Get token: https://hub.docker.com/settings/security
- Click "New Access Token"
- Copy and save as secret

## Workflow Triggers

- **Push to main/refactor/v2**: Runs tests → Builds → Pushes to Docker Hub
- **Pull request to main**: Runs tests only

## Docker Images

After successful build:
- `your-username/internship-recommender:latest`
- `your-username/internship-recommender:<commit-sha>`

## Pull and Run

```bash
docker pull your-username/internship-recommender:latest
docker run -p 8000:8000 your-username/internship-recommender:latest
```
