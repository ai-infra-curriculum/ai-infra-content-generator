# Lesson 08: API Development & Testing for ML

## Learning Objectives
- Design production-ready ML APIs
- Implement comprehensive error handling
- Write unit and integration tests
- Add monitoring and logging

## Duration: 3-4 hours

---

## 1. Production-Ready API Design

### 1.1 API Structure

```
/health          - Health check endpoint
/metrics         - Prometheus metrics
/v1/predict      - Main inference endpoint
/v1/batch        - Batch inference
/v1/models       - List available models
/v1/models/{id}  - Model information
```

### 1.2 Request/Response Format

**Request:**
```json
{
  "model_id": "resnet18-v1",
  "input": {
    "image_url": "https://example.com/cat.jpg"
  },
  "options": {
    "top_k": 5,
    "threshold": 0.5
  }
}
```

**Response:**
```json
{
  "predictions": [
    {"class": "cat", "probability": 0.95},
    {"class": "dog", "probability": 0.03}
  ],
  "model_id": "resnet18-v1",
  "inference_time_ms": 25,
  "request_id": "uuid-1234"
}
```

---

## 2. Robust Error Handling

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, validator
import logging

logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    image_url: str
    top_k: int = 5

    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError('top_k must be between 1 and 10')
        return v

@app.post("/v1/predict")
async def predict(request: PredictionRequest):
    try:
        # Download image
        image = download_image(request.image_url)

        # Run inference
        predictions = model.predict(image, top_k=request.top_k)

        logger.info(f"Successful prediction for {request.image_url}")
        return {"predictions": predictions}

    except ImageDownloadError as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download image: {str(e)}"
        )
    except ModelInferenceError as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed"
        )
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

---

## 3. Testing ML APIs

### 3.1 Unit Tests

```python
# tests/test_model.py
import pytest
import torch
from app.model import ModelWrapper

@pytest.fixture
def model():
    return ModelWrapper(model_path="models/resnet18.pth")

def test_model_loading(model):
    """Test model loads successfully"""
    assert model.model is not None
    assert model.device is not None

def test_inference_shape(model):
    """Test inference returns correct shape"""
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model.predict(input_tensor)
    assert output.shape[0] == batch_size

def test_prediction_probabilities(model):
    """Test predictions sum to 1"""
    input_tensor = torch.randn(1, 3, 224, 224)
    probs = model.predict_proba(input_tensor)
    assert pytest.approx(probs.sum().item(), 1.0)
```

### 3.2 Integration Tests

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test prediction endpoint with valid input"""
    response = client.post(
        "/v1/predict",
        json={
            "image_url": "https://example.com/cat.jpg",
            "top_k": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3

def test_predict_invalid_input():
    """Test prediction endpoint rejects invalid input"""
    response = client.post(
        "/v1/predict",
        json={"image_url": "not-a-url"}
    )
    assert response.status_code == 400
```

### 3.3 Load Testing

```python
# load_test.py
from locust import HttpUser, task, between

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        self.client.post(
            "/v1/predict",
            json={
                "image_url": "https://example.com/test.jpg",
                "top_k": 5
            }
        )

# Run: locust -f load_test.py --host=http://localhost:8000
```

---

## 4. Monitoring and Logging

### 4.1 Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        }
        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 4.2 Prometheus Metrics

```python
from prometheus_client import Counter, Histogram
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# Metrics
prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['model_id', 'status']
)
prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Prediction latency'
)

@app.post("/v1/predict")
@prediction_latency.time()
async def predict(request: PredictionRequest):
    try:
        result = await run_inference(request)
        prediction_counter.labels(
            model_id=request.model_id,
            status="success"
        ).inc()
        return result
    except Exception:
        prediction_counter.labels(
            model_id=request.model_id,
            status="error"
        ).inc()
        raise

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## 5. API Documentation

FastAPI auto-generates interactive docs:

```python
@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    summary="Run model inference",
    description="Submit image for classification",
    response_description="Predictions with probabilities"
)
async def predict(request: PredictionRequest):
    """
    Classify an image using the ML model.

    - **image_url**: URL of the image to classify
    - **top_k**: Number of top predictions to return (1-10)
    """
    pass
```

Visit `http://localhost:8000/docs` for interactive API docs.

---

## 6. Hands-On Exercise

**TODO: Build a production-ready ML API**

Requirements:
1. ✅ Input validation with Pydantic
2. ✅ Comprehensive error handling
3. ✅ Unit tests (80%+ coverage)
4. ✅ Integration tests
5. ✅ Structured logging
6. ✅ Prometheus metrics
7. ✅ API documentation
8. ✅ Load test (handle 100 req/sec)

**Starter template:**
```python
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI(title="Production ML API")

# TODO: Add middleware for logging
# TODO: Add Prometheus metrics
# TODO: Implement endpoints with error handling
# TODO: Add request validation
```

**Test suite:**
```bash
# Run tests
pytest tests/ -v --cov=app --cov-report=html

# Run load test
locust -f load_test.py --headless -u 100 -r 10 --run-time 60s
```

---

## 7. Key Takeaways

- ✅ Validate all inputs with Pydantic
- ✅ Handle errors gracefully with specific HTTP status codes
- ✅ Write comprehensive tests (unit + integration + load)
- ✅ Use structured logging for better debugging
- ✅ Add Prometheus metrics for monitoring
- ✅ Auto-generate API docs with FastAPI

---

## 8. Module 01 Completion

🎉 **Congratulations!** You've completed Module 01: Foundations

**What you've learned:**
- ML infrastructure fundamentals
- Cloud platforms for ML
- ML frameworks and model serving
- Docker containerization
- Production-ready API development

**Next steps:**
- ✅ Complete [Project 01: Basic Model Serving](../../projects/project-101-basic-model-serving/README.md)
- ✅ Proceed to Module 02: Cloud Computing
- ✅ Join discussions and ask questions!

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Locust Documentation](https://docs.locust.io/)
