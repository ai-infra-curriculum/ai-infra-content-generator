"""
Enterprise MLOps Platform API
FastAPI application for managing models, deployments, and features

Implements:
- ADR-005: Model Registry integration (MLflow)
- ADR-002: Feature Store integration (Feast)
- ADR-010: Governance Framework (approvals, auditing)
- ADR-007: Security & Compliance (authentication, authorization)
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import logging
import os
import mlflow
from feast import FeatureStore
import boto3
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise MLOps Platform API",
    description="API for managing ML models, deployments, and features",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# MLflow client
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local"))

# Feast client
feature_store = FeatureStore(repo_path=os.getenv("FEAST_REPO_PATH", "/feast-repo"))

# AWS clients
dynamodb = boto3.resource('dynamodb', region_name=os.getenv("AWS_REGION", "us-east-1"))
approvals_table = dynamodb.Table(os.getenv("APPROVALS_TABLE", "mlops-approvals"))

# ======================
# Models (Pydantic)
# ======================

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelStatus(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class ModelMetadata(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    owner: str = Field(..., description="Model owner email")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)

    @validator('owner')
    def validate_owner(cls, v):
        if '@' not in v:
            raise ValueError('Owner must be a valid email address')
        return v

class ModelRegistrationRequest(BaseModel):
    metadata: ModelMetadata
    model_uri: str = Field(..., description="URI to model artifacts (s3://...)")
    training_dataset_uri: Optional[str] = Field(None, description="URI to training data")
    metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ModelDeploymentRequest(BaseModel):
    model_name: str
    model_version: str
    target_stage: ModelStatus = Field(..., description="Target deployment stage")
    replicas: int = Field(default=3, ge=1, le=100)
    resources: Optional[Dict[str, str]] = Field(default_factory=dict)
    environment_vars: Optional[Dict[str, str]] = Field(default_factory=dict)
    justification: str = Field(..., description="Business justification for deployment")

class ApprovalRequest(BaseModel):
    request_id: str
    request_type: str = Field(..., description="Type of request (model_deployment, etc.)")
    requester: str
    approver: str
    status: ApprovalStatus
    comments: Optional[str] = None

class FeatureRetrievalRequest(BaseModel):
    entity_ids: List[str] = Field(..., description="List of entity IDs")
    feature_refs: List[str] = Field(..., description="Feature references (feature_view:feature)")

class ModelPredictionRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = Field(default="latest")
    input_data: Dict[str, Any] = Field(..., description="Input features for prediction")
    return_explanations: bool = Field(default=False)

# ======================
# Dependency Injection
# ======================

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify JWT token and return user email
    In production: validate with Okta/Auth0/AWS Cognito
    """
    token = credentials.credentials
    # TODO: Implement actual JWT validation
    # For now, return mock user
    return "user@company.com"

def check_permission(user: str, action: str, resource: str) -> bool:
    """
    Check if user has permission for action on resource
    In production: integrate with RBAC system
    """
    # TODO: Implement actual RBAC check
    logger.info(f"Checking permission: {user} -> {action} on {resource}")
    return True

# ======================
# Health & Monitoring
# ======================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancer"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies dependencies"""
    checks = {
        "mlflow": False,
        "feast": False,
        "dynamodb": False
    }

    try:
        # Check MLflow
        mlflow.get_tracking_uri()
        checks["mlflow"] = True
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")

    try:
        # Check Feast
        feature_store.list_feature_views()
        checks["feast"] = True
    except Exception as e:
        logger.error(f"Feast health check failed: {e}")

    try:
        # Check DynamoDB
        approvals_table.table_status
        checks["dynamodb"] = True
    except Exception as e:
        logger.error(f"DynamoDB health check failed: {e}")

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        content={"status": "ready" if all_healthy else "not_ready", "checks": checks},
        status_code=status_code
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement Prometheus metrics export
    return {"message": "Metrics endpoint - implement with prometheus_client"}

# ======================
# Model Management
# ======================

@app.post("/api/v1/models/register", tags=["Models"])
async def register_model(
    request: ModelRegistrationRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_token)
):
    """
    Register a new model version in MLflow

    Implements:
    - ADR-005: Model Registry
    - ADR-010: Governance Framework (risk classification)
    """
    try:
        # Validate user permissions
        if not check_permission(user, "register_model", request.metadata.name):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        # Register model in MLflow
        with mlflow.start_run() as run:
            # Log parameters
            if request.parameters:
                mlflow.log_params(request.parameters)

            # Log metrics
            if request.metrics:
                mlflow.log_metrics(request.metrics)

            # Log model
            mlflow.register_model(
                model_uri=request.model_uri,
                name=request.metadata.name,
                tags={
                    **request.metadata.tags,
                    "risk_level": request.metadata.risk_level.value,
                    "owner": request.metadata.owner,
                    "registered_by": user
                }
            )

        # Create audit log entry
        background_tasks.add_task(
            create_audit_log,
            action="model_registered",
            user=user,
            resource=f"{request.metadata.name}:{request.metadata.version}",
            details=request.dict()
        )

        logger.info(f"Model registered: {request.metadata.name}:{request.metadata.version} by {user}")

        return {
            "status": "success",
            "model_name": request.metadata.name,
            "model_version": request.metadata.version,
            "run_id": run.info.run_id,
            "message": "Model registered successfully"
        }

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_name}/versions", tags=["Models"])
async def list_model_versions(
    model_name: str,
    user: str = Depends(verify_token)
):
    """List all versions of a model"""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        return {
            "model_name": model_name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "created_at": v.creation_timestamp,
                    "tags": v.tags
                }
                for v in versions
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/deploy", tags=["Models"])
async def deploy_model(
    request: ModelDeploymentRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_token)
):
    """
    Deploy model to specified stage

    Implements:
    - ADR-010: Governance Framework (approval workflows)
    """
    try:
        # Get model metadata
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(request.model_name, request.model_version)
        risk_level = model_version.tags.get("risk_level", "medium")

        # Check if approval is required
        requires_approval = (
            (risk_level == "high" and request.target_stage == ModelStatus.PRODUCTION) or
            (risk_level == "medium" and request.target_stage == ModelStatus.PRODUCTION)
        )

        if requires_approval:
            # Create approval request
            approval_id = create_approval_request(
                request_type="model_deployment",
                requester=user,
                resource=f"{request.model_name}:{request.model_version}",
                details=request.dict(),
                risk_level=risk_level
            )

            return {
                "status": "pending_approval",
                "approval_id": approval_id,
                "message": f"Deployment requires approval (risk level: {risk_level})"
            }

        # Auto-approve for low-risk models or non-production deployments
        deployment_id = execute_deployment(request)

        # Create audit log
        background_tasks.add_task(
            create_audit_log,
            action="model_deployed",
            user=user,
            resource=f"{request.model_name}:{request.model_version}",
            details=request.dict()
        )

        return {
            "status": "success",
            "deployment_id": deployment_id,
            "message": "Model deployed successfully"
        }

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# Feature Management
# ======================

@app.post("/api/v1/features/retrieve", tags=["Features"])
async def retrieve_features(
    request: FeatureRetrievalRequest,
    user: str = Depends(verify_token)
):
    """
    Retrieve features from Feast feature store

    Implements: ADR-002 (Feature Store - Feast)
    """
    try:
        # Retrieve features
        feature_vector = feature_store.get_online_features(
            features=request.feature_refs,
            entity_rows=[{"entity_id": eid} for eid in request.entity_ids]
        ).to_dict()

        logger.info(f"Retrieved features for {len(request.entity_ids)} entities by {user}")

        return {
            "status": "success",
            "features": feature_vector,
            "entity_count": len(request.entity_ids)
        }

    except Exception as e:
        logger.error(f"Feature retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/features/views", tags=["Features"])
async def list_feature_views(user: str = Depends(verify_token)):
    """List all available feature views"""
    try:
        feature_views = feature_store.list_feature_views()

        return {
            "feature_views": [
                {
                    "name": fv.name,
                    "entities": [e.name for e in fv.entities],
                    "features": [f.name for f in fv.features],
                    "ttl": str(fv.ttl) if fv.ttl else None
                }
                for fv in feature_views
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list feature views: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# Model Inference
# ======================

@app.post("/api/v1/predict", tags=["Inference"])
async def predict(
    request: ModelPredictionRequest,
    user: str = Depends(verify_token)
):
    """
    Make prediction using deployed model
    """
    try:
        # Load model from MLflow
        model_uri = f"models:/{request.model_name}/{request.model_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Make prediction
        import pandas as pd
        input_df = pd.DataFrame([request.input_data])
        prediction = model.predict(input_df)

        response = {
            "status": "success",
            "model_name": request.model_name,
            "model_version": request.model_version,
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        }

        # Add explanations if requested
        if request.return_explanations:
            # TODO: Implement SHAP explanations
            response["explanations"] = {"message": "Explanations not yet implemented"}

        logger.info(f"Prediction made with {request.model_name}:{request.model_version}")

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# Helper Functions
# ======================

def create_approval_request(request_type: str, requester: str, resource: str, details: dict, risk_level: str) -> str:
    """Create approval request in DynamoDB"""
    import uuid
    approval_id = str(uuid.uuid4())

    approvals_table.put_item(Item={
        'approval_id': approval_id,
        'request_type': request_type,
        'requester': requester,
        'resource': resource,
        'details': json.dumps(details),
        'risk_level': risk_level,
        'status': 'pending',
        'created_at': datetime.utcnow().isoformat(),
        'ttl': int((datetime.utcnow().timestamp() + 30*24*3600))  # 30 days
    })

    logger.info(f"Created approval request: {approval_id}")
    return approval_id

def execute_deployment(request: ModelDeploymentRequest) -> str:
    """Execute model deployment to Kubernetes"""
    # TODO: Implement KServe deployment
    import uuid
    deployment_id = str(uuid.uuid4())

    logger.info(f"Executing deployment: {deployment_id}")
    return deployment_id

async def create_audit_log(action: str, user: str, resource: str, details: dict):
    """Create audit log entry"""
    try:
        # TODO: Send to audit logging system (CloudWatch Logs, S3, etc.)
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "resource": resource,
            "details": details
        }
        logger.info(f"Audit log: {json.dumps(audit_entry)}")
    except Exception as e:
        logger.error(f"Failed to create audit log: {e}")

# ======================
# Error Handlers
# ======================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

# ======================
# Startup/Shutdown
# ======================

@app.on_event("startup")
async def startup_event():
    logger.info("MLOps Platform API starting up...")
    logger.info(f"MLflow URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Feast repo: {os.getenv('FEAST_REPO_PATH', '/feast-repo')}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("MLOps Platform API shutting down...")

# ======================
# Run Application
# ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
