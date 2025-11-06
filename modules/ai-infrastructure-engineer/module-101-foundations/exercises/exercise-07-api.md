# Exercise 07: FastAPI ML Service

**Duration:** 4-5 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 08 (API Development), Exercise 02 (Docker)

## Learning Objectives

By completing this exercise, you will:
- Build a production-ready FastAPI application for ML model serving
- Implement request/response validation with Pydantic
- Add health checks and monitoring endpoints
- Handle errors gracefully
- Write API tests with pytest
- Document APIs with auto-generated OpenAPI specs
- Deploy the service with Docker

---

## Overview

In this exercise, you'll build a complete FastAPI service that serves a machine learning model. You'll implement best practices for production ML APIs including validation, error handling, logging, and testing.

We'll use a simple pre-trained image classification model (MobileNetV2) from TorchVision as an example, but the patterns you learn apply to any ML model.

---

## Part 1: Project Setup (30 minutes)

### Task 1.1: Create Project Structure

```bash
mkdir fastapi-ml-service && cd fastapi-ml-service
mkdir -p app/{api,models,core} tests models
touch app/__init__.py app/main.py
touch app/api/__init__.py app/api/endpoints.py
touch app/models/__init__.py app/models/schemas.py app/models/ml_model.py
touch app/core/__init__.py app/core/config.py app/core/logging.py
touch tests/__init__.py tests/test_api.py tests/conftest.py
touch requirements.txt Dockerfile .dockerignore README.md
```

Your structure should look like:
```
fastapi-ml-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py     # API routes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py       # Pydantic models
│   │   └── ml_model.py      # ML model wrapper
│   └── core/
│       ├── __init__.py
│       ├── config.py        # Configuration
│       └── logging.py       # Logging setup
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── conftest.py
├── models/                   # Model files
├── requirements.txt
├── Dockerfile
└── README.md
```

### Task 1.2: Create Requirements File

Create `requirements.txt`:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
numpy==1.26.2
```

Create `requirements-dev.txt`:
```
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.1
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Checkpoint:** All dependencies installed successfully

---

## Part 2: ML Model Wrapper (45 minutes)

### Task 2.1: Create Configuration

Create `app/core/config.py`:
```python
"""Application configuration"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # App metadata
    APP_NAME: str = "FastAPI ML Service"
    APP_VERSION: str = "0.1.0"

    # API settings
    API_V1_PREFIX: str = "/api/v1"

    # Model settings
    MODEL_PATH: str = "models/mobilenet_v2.pth"
    MODEL_DEVICE: str = "cpu"  # or "cuda" if GPU available

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Logging
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

### Task 2.2: Implement Model Wrapper

Create `app/models/ml_model.py`:
```python
"""ML Model wrapper for inference"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    Wrapper for image classification model

    Uses MobileNetV2 pre-trained on ImageNet
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Initialize the classifier

        Args:
            model_path: Path to model weights (optional, uses pretrained if None)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self.class_labels = None
        self._load_model(model_path)
        self._setup_transforms()
        self._load_imagenet_labels()

    def _load_model(self, model_path: str = None):
        """Load model architecture and weights"""
        logger.info("Loading MobileNetV2 model...")

        # Load pre-trained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)

        # If custom weights provided, load them
        if model_path and Path(model_path).exists():
            logger.info(f"Loading custom weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        # Simplified - in production, load from file
        self.class_labels = [f"class_{i}" for i in range(1000)]
        # TODO: Load actual ImageNet labels from file
        logger.info("Loaded 1000 ImageNet class labels")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image for model input

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed image tensor
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Apply transforms
            image_tensor = self.transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")

    def predict(self, image_bytes: bytes, top_k: int = 5) -> Dict[str, Any]:
        """
        Run inference on image

        Args:
            image_bytes: Raw image bytes
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and metadata
        """
        import time
        start_time = time.time()

        try:
            # Preprocess
            image_tensor = self.preprocess_image(image_bytes)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Format results
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    "class": self.class_labels[idx.item()],
                    "class_id": idx.item(),
                    "confidence": float(prob.item())
                })

            inference_time = time.time() - start_time

            return {
                "predictions": predictions,
                "top_prediction": predictions[0],
                "metadata": {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model": "MobileNetV2",
                    "device": str(self.device)
                }
            }

        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Model inference failed: {str(e)}")


# Global model instance (loaded once at startup)
_model_instance = None


def get_model() -> ImageClassifier:
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None:
        from app.core.config import settings
        _model_instance = ImageClassifier(
            model_path=settings.MODEL_PATH,
            device=settings.MODEL_DEVICE
        )
    return _model_instance
```

**Important:** Add missing import at the top:
```python
import io
```

**Checkpoint:** Model wrapper loads and can accept image bytes

---

## Part 3: Pydantic Schemas (30 minutes)

### Task 3.1: Define Request/Response Models

Create `app/models/schemas.py`:
```python
"""Pydantic schemas for request/response validation"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="List of predictions with confidence scores"
    )
    top_prediction: Dict[str, Any] = Field(
        ...,
        description="Top prediction"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Inference metadata (time, model, device)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "class": "golden_retriever",
                        "class_id": 207,
                        "confidence": 0.92
                    },
                    {
                        "class": "labrador",
                        "class_id": 208,
                        "confidence": 0.05
                    }
                ],
                "top_prediction": {
                    "class": "golden_retriever",
                    "class_id": 207,
                    "confidence": 0.92
                },
                "metadata": {
                    "inference_time_ms": 45.2,
                    "model": "MobileNetV2",
                    "device": "cpu"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "model_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""

    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid image format",
                "error_type": "ValidationError"
            }
        }
```

**Checkpoint:** Schemas defined with proper validation and examples

---

## Part 4: API Endpoints (60 minutes)

### Task 4.1: Create API Routes

Create `app/api/endpoints.py`:
```python
"""API endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.models.schemas import PredictionResponse, HealthResponse, ErrorResponse
from app.models.ml_model import get_model
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the service is running and model is loaded"
)
async def health_check():
    """
    Health check endpoint

    Returns service status and model availability
    """
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version=settings.APP_VERSION,
        model_loaded=model_loaded
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Predictions"],
    summary="Classify image",
    description="Upload an image and get top-5 class predictions"
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)")
):
    """
    Image classification endpoint

    Accepts an image file and returns top-5 predictions with confidence scores

    Args:
        file: Uploaded image file

    Returns:
        PredictionResponse with predictions and metadata

    Raises:
        HTTPException: If image is invalid or prediction fails
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG or PNG"
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Validate file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 10MB"
            )

        # Get model and run prediction
        model = get_model()
        result = model.predict(image_bytes, top_k=5)

        logger.info(
            f"Prediction completed: {result['top_prediction']['class']} "
            f"({result['top_prediction']['confidence']:.2f}) "
            f"in {result['metadata']['inference_time_ms']}ms"
        )

        return PredictionResponse(**result)

    except ValueError as e:
        # Validation errors (bad image format)
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        # Internal errors
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )
```

### Task 4.2: Create Main Application

Create `app/main.py`:
```python
"""FastAPI ML Service - Main Application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import endpoints
from app.core.config import settings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready ML model serving with FastAPI",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(
    endpoints.router,
    prefix=settings.API_V1_PREFIX,
    tags=["api"]
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health"
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("Loading ML model...")

    try:
        from app.models.ml_model import get_model
        model = get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down FastAPI ML Service")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL
    )
```

**Checkpoint:** API runs locally with `python app/main.py`

---

## Part 5: Testing (45 minutes)

### Task 5.1: Create Test Fixtures

Create `tests/conftest.py`:
```python
"""Pytest fixtures for API tests"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    image = Image.new('RGB', (224, 224), color='red')

    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes


@pytest.fixture
def invalid_image():
    """Create invalid image data"""
    return io.BytesIO(b"not an image")
```

### Task 5.2: Write API Tests

Create `tests/test_api.py`:
```python
"""API endpoint tests"""

import pytest
from fastapi import status


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "version" in data
    assert "model_loaded" in data


def test_predict_with_valid_image(client, sample_image):
    """Test prediction with valid image"""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Check response structure
    assert "predictions" in data
    assert "top_prediction" in data
    assert "metadata" in data

    # Check predictions
    assert len(data["predictions"]) == 5
    assert all("class" in p for p in data["predictions"])
    assert all("confidence" in p for p in data["predictions"])

    # Check metadata
    assert "inference_time_ms" in data["metadata"]
    assert "model" in data["metadata"]


def test_predict_with_invalid_content_type(client, sample_image):
    """Test prediction with invalid content type"""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", sample_image, "text/plain")}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_predict_with_invalid_image(client, invalid_image):
    """Test prediction with corrupted image"""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", invalid_image, "image/jpeg")}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_predict_without_file(client):
    """Test prediction without uploading file"""
    response = client.post("/api/v1/predict")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_prediction_response_time(client, sample_image):
    """Test that prediction completes in reasonable time"""
    import time

    start = time.time()
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )
    duration = time.time() - start

    assert response.status_code == status.HTTP_200_OK
    assert duration < 5.0  # Should complete in < 5 seconds on CPU
```

### Task 5.3: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Expected Output:**
```
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_predict_with_valid_image PASSED
tests/test_api.py::test_predict_with_invalid_content_type PASSED
tests/test_api.py::test_predict_with_invalid_image PASSED
tests/test_api.py::test_predict_without_file PASSED
tests/test_api.py::test_prediction_response_time PASSED

========== 7 passed in 3.45s ==========
```

**Checkpoint:** All tests passing

---

## Part 6: Docker Deployment (30 minutes)

### Task 6.1: Create Dockerfile

Create `Dockerfile`:
```dockerfile
# Multi-stage build for FastAPI ML Service

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app/ /app/app/

# Switch to non-root user
USER appuser

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `.dockerignore`:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
.coverage
htmlcov/
*.log
.DS_Store
.git
.gitignore
README.md
tests/
```

### Task 6.2: Build and Run Docker Container

```bash
# Build image
docker build -t fastapi-ml-service:latest .

# Run container
docker run -d \
    -p 8000:8000 \
    --name ml-api \
    fastapi-ml-service:latest

# Check logs
docker logs ml-api

# Test health endpoint
curl http://localhost:8000/api/v1/health

# Stop container
docker stop ml-api
docker rm ml-api
```

**Checkpoint:** Docker container builds and runs successfully

---

## Part 7: Testing the API (15 minutes)

### Task 7.1: Test with cURL

```bash
# Start the service
python app/main.py

# In another terminal:

# Test root endpoint
curl http://localhost:8000/

# Test health check
curl http://localhost:8000/api/v1/health

# Test prediction with an image
curl -X POST \
    -F "file=@/path/to/your/image.jpg" \
    http://localhost:8000/api/v1/predict

# View API documentation
open http://localhost:8000/docs
```

### Task 7.2: Test with Python

Create `test_client.py`:
```python
"""Simple client for testing the API"""

import requests


def test_health():
    """Test health endpoint"""
    response = requests.get("http://localhost:8000/api/v1/health")
    print("Health Check:", response.json())


def test_prediction(image_path: str):
    """Test prediction endpoint"""
    with open(image_path, "rb") as f:
        files = {"file": ("image.jpg", f, "image/jpeg")}
        response = requests.post(
            "http://localhost:8000/api/v1/predict",
            files=files
        )

    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Results:")
        print(f"Top prediction: {result['top_prediction']['class']}")
        print(f"Confidence: {result['top_prediction']['confidence']:.2%}")
        print(f"Inference time: {result['metadata']['inference_time_ms']}ms")

        print("\nTop 5 predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    test_health()

    # Test with your own image
    # test_prediction("path/to/your/image.jpg")
```

**Checkpoint:** API responds correctly to all test requests

---

## Part 8: Documentation (15 minutes)

### Task 8.1: Create README

Create `README.md`:
```markdown
# FastAPI ML Service

Production-ready image classification API built with FastAPI and PyTorch.

## Features

- ✅ FastAPI with async support
- ✅ Pre-trained MobileNetV2 model
- ✅ Pydantic validation
- ✅ Auto-generated OpenAPI docs
- ✅ Comprehensive error handling
- ✅ Unit tests with pytest
- ✅ Docker containerization
- ✅ Health check endpoint

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run the service**:
   ```bash
   python app/main.py
   ```

3. **Open API docs**: http://localhost:8000/docs

### Docker

```bash
# Build
docker build -t fastapi-ml-service .

# Run
docker run -p 8000:8000 fastapi-ml-service
```

## API Endpoints

### GET /api/v1/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true
}
```

### POST /api/v1/predict

Classify an image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file (JPEG/PNG)

**Response:**
```json
{
  "predictions": [
    {
      "class": "golden_retriever",
      "class_id": 207,
      "confidence": 0.92
    }
  ],
  "top_prediction": {
    "class": "golden_retriever",
    "class_id": 207,
    "confidence": 0.92
  },
  "metadata": {
    "inference_time_ms": 45.2,
    "model": "MobileNetV2",
    "device": "cpu"
  }
}
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## Project Structure

```
fastapi-ml-service/
├── app/
│   ├── main.py           # FastAPI application
│   ├── api/
│   │   └── endpoints.py  # API routes
│   ├── models/
│   │   ├── schemas.py    # Pydantic models
│   │   └── ml_model.py   # ML model wrapper
│   └── core/
│       └── config.py     # Configuration
├── tests/                # Unit tests
├── Dockerfile
└── requirements.txt
```

## Configuration

Edit `app/core/config.py` or set environment variables:

- `MODEL_PATH`: Path to model weights
- `MODEL_DEVICE`: "cpu" or "cuda"
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: "info")

## License

MIT
```

**Checkpoint:** Documentation complete

---

## Deliverables

Submit a GitHub repository with:

1. ✅ **Complete FastAPI application**
   - `app/main.py` with FastAPI setup
   - `app/api/endpoints.py` with routes
   - `app/models/schemas.py` with Pydantic models
   - `app/models/ml_model.py` with model wrapper

2. ✅ **Tests** with pytest
   - `tests/test_api.py` with endpoint tests
   - `tests/conftest.py` with fixtures
   - All tests passing

3. ✅ **Docker configuration**
   - `Dockerfile` with multi-stage build
   - `.dockerignore` file
   - Container builds and runs

4. ✅ **Documentation**
   - `README.md` with setup and usage
   - Auto-generated OpenAPI docs at `/docs`

---

## Success Criteria

- [ ] FastAPI application runs locally
- [ ] Health check endpoint returns 200
- [ ] Prediction endpoint accepts images and returns classifications
- [ ] All unit tests pass (7/7)
- [ ] Test coverage > 80%
- [ ] Docker container builds successfully
- [ ] API documentation accessible at `/docs`
- [ ] Proper error handling for invalid inputs
- [ ] Response time < 5 seconds on CPU

---

## Bonus Challenges

1. **Add batch prediction** endpoint that accepts multiple images
2. **Implement caching** for repeated predictions
3. **Add rate limiting** with slowapi
4. **Add authentication** with API keys
5. **Deploy to cloud** (AWS, GCP, or Azure)
6. **Add Prometheus metrics** for monitoring
7. **Implement model versioning** to serve multiple models

---

## Common Issues and Solutions

### Issue: Model download fails
**Solution:** MobileNetV2 downloads automatically on first run. Ensure internet connection.

### Issue: "Model not loaded" error
**Solution:** Check logs for model loading errors. Verify PyTorch installation.

### Issue: Slow predictions
**Solution:** First inference is slower due to model initialization. Subsequent calls are faster.

### Issue: Docker build fails
**Solution:** Ensure Docker has sufficient memory (recommend 4GB+).

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Estimated Time:** 4-5 hours
**Next Exercise:** Exercise 04 (Advanced Python Environment Manager)
