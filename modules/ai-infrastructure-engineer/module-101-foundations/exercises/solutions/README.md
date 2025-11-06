# Module 01: Foundations - Exercise Solutions

This directory contains complete solutions for Module 01 exercises.

## 📌 Important Note

**Try to solve exercises on your own first!** These solutions are provided to:
- Check your work after completing exercises
- Understand alternative approaches
- Learn best practices
- Debug issues in your implementation

## 🎯 Available Solutions

### Python Exercises
1. **Environment Setup Solution** - Virtual environment and dependency management
2. **FastAPI Basic API Solution** - Simple REST API implementation
3. **Docker Basics Solution** - Containerizing a Python application

### Git Exercises
4. **Git Workflow Solution** - Branching, merging, and collaboration
5. **Git Best Practices Solution** - Gitignore, commit messages, etc.

### Linux Exercises
6. **Command Line Solution** - Essential Linux commands
7. **Shell Scripting Solution** - Automation scripts

## 📁 Solution Files

```
solutions/
├── README.md (this file)
├── 01-python-environment.md
├── 02-fastapi-basic.py
├── 03-docker-basics/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── 04-git-workflow.md
├── 05-git-best-practices.md
├── 06-linux-commands.md
└── 07-shell-scripting.sh
```

---

## Solution 1: Python Environment Setup

**Exercise:** Create a virtual environment and install dependencies

### Solution Steps

```bash
# 1. Create virtual environment
python3 -m venv ml-env

# 2. Activate environment
source ml-env/bin/activate  # Linux/Mac
# ml-env\Scripts\activate  # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install numpy pandas scikit-learn fastapi uvicorn

# 5. Freeze requirements
pip freeze > requirements.txt

# 6. Verify installation
python -c "import numpy; import pandas; import sklearn; print('All packages installed!')"
```

### Best Practices
- Always use virtual environments (never install globally)
- Pin package versions in requirements.txt
- Use `pip freeze` to capture exact versions
- Document Python version needed

---

## Solution 2: FastAPI Basic API

**Exercise:** Create a simple ML prediction API

### Complete Solution

```python
# solutions/02-fastapi-basic.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Prediction API",
    description="Simple API for model predictions",
    version="1.0.0"
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float] = Field(..., min_items=1, description="Input features")

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    model_version: str = "1.0"

# Mock model (replace with actual model)
def predict(features: List[float]) -> float:
    """
    Simple prediction function (mock).
    Replace with actual model inference.
    """
    return sum(features) / len(features)  # Average as mock prediction

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Make a prediction based on input features.

    Args:
        request: PredictionRequest with features

    Returns:
        PredictionResponse with prediction
    """
    try:
        logger.info(f"Received prediction request with {len(request.features)} features")

        # Validate input
        if len(request.features) == 0:
            raise HTTPException(status_code=400, detail="Features cannot be empty")

        # Make prediction
        prediction = predict(request.features)

        logger.info(f"Prediction successful: {prediction}")

        return PredictionResponse(
            prediction=prediction,
            model_version="1.0"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the Solution

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Run server
python solutions/02-fastapi-basic.py

# Test in another terminal
curl http://localhost:8000/
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

### Key Learnings
- Use Pydantic for request/response validation
- Add proper error handling
- Include logging for debugging
- Document API with docstrings
- Provide example requests in schema

---

## Solution 3: Docker Basics

**Exercise:** Containerize the FastAPI application

### Dockerfile
```dockerfile
# solutions/03-docker-basics/Dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### requirements.txt
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

### .dockerignore
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
.dockerignore
Dockerfile
README.md
.pytest_cache
.coverage
```

### Build and Run Commands
```bash
# Build image
docker build -t ml-api:v1 solutions/03-docker-basics/

# Run container
docker run -d -p 8000:8000 --name ml-api ml-api:v1

# Test
curl http://localhost:8000/health

# View logs
docker logs ml-api

# Stop and remove
docker stop ml-api
docker rm ml-api
```

### Docker Best Practices Demonstrated
✅ Multi-stage build (reduces image size)
✅ Non-root user (security)
✅ .dockerignore (excludes unnecessary files)
✅ Health check (for orchestration)
✅ Specific package versions (reproducibility)
✅ No cache during pip install (smaller image)

---

## Solution 4: Git Workflow

**Exercise:** Practice branching, merging, and collaboration

### Solution Steps

```bash
# 1. Create new feature branch
git checkout -b feature/add-logging

# 2. Make changes (add logging to code)
# Edit files...

# 3. Stage and commit changes
git add app.py
git commit -m "Add structured logging to prediction endpoint

- Added Python logging configuration
- Log all prediction requests with details
- Log errors with full context
- Follows logging best practices"

# 4. Push branch to remote
git push -u origin feature/add-logging

# 5. Create pull request (on GitHub)
# Go to repository and create PR

# 6. After PR approval, merge to main
git checkout main
git pull origin main
git merge feature/add-logging

# 7. Delete feature branch
git branch -d feature/add-logging
git push origin --delete feature/add-logging

# 8. Tag release
git tag -a v1.1.0 -m "Version 1.1.0 - Added logging"
git push origin v1.1.0
```

### Commit Message Best Practices
- Use imperative mood ("Add" not "Added")
- First line is summary (< 50 chars)
- Blank line, then detailed description
- Explain WHY, not WHAT
- Reference issues if applicable

---

## Solution 5: Git Best Practices

**Exercise:** Set up proper .gitignore and commit practices

### .gitignore for Python/ML Project
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# ML/Data
*.h5
*.pkl
*.joblib
*.onnx
*.pt
*.pth
models/
data/
datasets/

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local
*.env

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
dist/
build/
*.egg-info/
```

### Pre-commit Hook Example
```bash
# .git/hooks/pre-commit
#!/bin/sh

# Run tests before commit
pytest tests/

# Run linter
flake8 src/

# Check for secrets
if git diff --cached | grep -i 'api_key\|password\|secret'; then
    echo "Warning: Possible secret in commit!"
    exit 1
fi
```

---

## Additional Resources

### Documentation Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

### Next Steps
1. Try implementing variations of these solutions
2. Add tests for the FastAPI application
3. Set up CI/CD for Docker builds
4. Experiment with advanced Git features

---

## Questions?

If you have questions about these solutions:
1. Review the relevant lesson material
2. Check the documentation links
3. Ask in [GitHub Discussions](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/discussions)

**Remember:** Understanding WHY these solutions work is more important than just copying the code!
