# Lesson 03: ML Infrastructure Basics

**Duration:** 4 hours

## Learning Objectives

By the end of this lecture, you will be able to:
- Describe the complete ML lifecycle from data collection to production deployment
- Identify infrastructure requirements for each stage of the ML lifecycle
- Explain the differences between training infrastructure and serving infrastructure
- Understand data versioning and lineage tracking requirements
- Describe model versioning, registry, and deployment strategies
- Identify monitoring and observability requirements for ML systems
- Explain the concept of feedback loops in production ML systems

## Introduction

Machine learning infrastructure exists to support the complete ML lifecycle - from data collection through model deployment and monitoring. Understanding this lifecycle and the infrastructure requirements at each stage is fundamental to being an effective ML infrastructure engineer.

## The ML Lifecycle

The ML lifecycle consists of several interconnected stages that form a continuous loop:

```
┌──────────────────────────────────────────────────────────────────┐
│                     ML Lifecycle                                 │
└──────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │   1. DATA   │
  │ Collection  │
  │ Preparation │
  │  Labeling   │
  └──────┬──────┘
         │
         ↓
  ┌─────────────┐
  │ 2. TRAINING │
  │Experiment   │
  │Development  │
  │Validation   │
  └──────┬──────┘
         │
         ↓
  ┌─────────────┐
  │3. EVALUATION│
  │   Testing   │
  │ Validation  │
  │   Metrics   │
  └──────┬──────┘
         │
         ↓
  ┌─────────────┐
  │4. DEPLOYMENT│
  │  Serving    │
  │Integration  │
  │   Scaling   │
  └──────┬──────┘
         │
         ↓
  ┌─────────────┐
  │5. MONITORING│
  │Performance  │
  │  Metrics    │
  │ Drift Det.  │
  └──────┬──────┘
         │
         ↓
  ┌─────────────┐
  │6. RETRAINING│
  │Update Model │
  │  New Data   │
  │   Deploy    │
  └──────┬──────┘
         │
         └────────────┐
                      │
                      └────> (Loop back to Data)
```

## Stage 1: Data Collection and Preparation

### What Happens
- Raw data is collected from various sources (databases, APIs, sensors, logs)
- Data is cleaned, validated, and transformed
- Features are engineered from raw data
- Data is labeled (for supervised learning)
- Datasets are versioned and stored

### Infrastructure Requirements

**Storage:**
- **Object storage** (S3, GCS, Azure Blob) for large datasets
- **Data lakes** for raw, unstructured data
- **Data warehouses** for structured, processed data
- **Version control for data** (DVC, Delta Lake)

**Compute:**
- **Data processing clusters** (Spark, Dask) for large-scale transformations
- **ETL/ELT pipelines** for data movement and transformation
- **Distributed computing** for parallel processing

**Tools:**
- **Apache Airflow** - Workflow orchestration
- **Apache Kafka** - Real-time data streaming
- **DVC** - Data version control
- **Great Expectations** - Data validation

### Example Infrastructure

```python
# Data pipeline orchestration with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def collect_data():
    # Collect from API
    pass

def validate_data():
    # Check data quality
    pass

def transform_data():
    # Feature engineering
    pass

dag = DAG('data_pipeline', start_date=datetime(2025, 1, 1))

collect = PythonOperator(task_id='collect', python_callable=collect_data, dag=dag)
validate = PythonOperator(task_id='validate', python_callable=validate_data, dag=dag)
transform = PythonOperator(task_id='transform', python_callable=transform_data, dag=dag)

collect >> validate >> transform
```

## Stage 2: Model Training and Experimentation

### What Happens
- Data scientists experiment with different algorithms
- Models are trained on prepared datasets
- Hyperparameters are tuned
- Multiple experiments are run and tracked
- Best models are selected based on validation metrics

### Infrastructure Requirements

**Compute:**
- **GPU clusters** for deep learning training
- **Distributed training** across multiple GPUs/nodes
- **Resource scheduling** (Kubernetes, SLURM)
- **Spot instances** for cost optimization

**Experiment Tracking:**
- **MLflow** - Experiment tracking and model registry
- **Weights & Biases** - Experiment visualization
- **TensorBoard** - Training metrics visualization

**Storage:**
- **Model registry** for versioning trained models
- **Artifact storage** for model files, checkpoints
- **Metadata storage** for experiment parameters

### Example Infrastructure

```python
# Experiment tracking with MLflow
import mlflow
import torch

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)

# Training loop
for epoch in range(num_epochs):
    loss = train_epoch(model, data_loader)

    # Log metrics
    mlflow.log_metric("loss", loss, step=epoch)
    mlflow.log_metric("accuracy", accuracy, step=epoch)

# Save model
mlflow.pytorch.log_model(model, "model")

mlflow.end_run()
```

### Resource Requirements

| Model Type | GPU Memory | Training Time | Storage |
|-----------|------------|---------------|---------|
| Small CNN | 4-8 GB | Hours | <1 GB |
| Large CNN (ResNet-152) | 12-16 GB | Days | 5-10 GB |
| BERT Base | 16-24 GB | Days | 500 MB |
| GPT-3 Scale | 80 GB+ (multiple) | Weeks | 100s of GB |
| LLM (7B params) | 28 GB+ | Days-Weeks | 14-28 GB |

## Stage 3: Model Evaluation and Validation

### What Happens
- Models are evaluated on test datasets
- Performance metrics are calculated
- Model behavior is analyzed (confusion matrices, error analysis)
- Edge cases and failure modes are identified
- Model fairness and bias are assessed

### Infrastructure Requirements

**Compute:**
- Inference infrastructure for batch evaluation
- A/B testing frameworks

**Tools:**
- **Model evaluation frameworks** (scikit-learn metrics, custom evaluators)
- **Bias detection tools** (Fairlearn, AI Fairness 360)
- **Model explainability** (SHAP, LIME)

### Key Metrics to Track

**Classification:**
- Accuracy, Precision, Recall, F1 Score
- ROC AUC, Precision-Recall AUC
- Confusion Matrix

**Regression:**
- MSE, RMSE, MAE
- R² Score
- Residual plots

**Ranking:**
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

**Business Metrics:**
- Revenue impact
- User engagement
- Conversion rate

## Stage 4: Model Deployment

### What Happens
- Model is packaged for production
- Model is deployed to serving infrastructure
- API endpoints are created for inference
- Load balancing and scaling are configured
- Gradual rollout (canary deployment)

### Infrastructure Requirements

**Serving:**
- **Model serving frameworks** (TorchServe, TensorFlow Serving, FastAPI)
- **Container orchestration** (Kubernetes)
- **Load balancers** for traffic distribution
- **API gateways** for request routing

**Deployment Strategies:**

1. **Blue-Green Deployment**
   - Run two identical production environments
   - Switch traffic from old (blue) to new (green)
   - Instant rollback if issues

2. **Canary Deployment**
   - Gradually route traffic to new version
   - Monitor metrics closely
   - Roll back if degradation detected

3. **Shadow Deployment**
   - Send traffic to both old and new models
   - Compare predictions without affecting users
   - Validate before full deployment

### Example Deployment Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ↓              ↓              ↓
         ┌────────┐     ┌────────┐     ┌────────┐
         │ Model  │     │ Model  │     │ Model  │
         │ Pod 1  │     │ Pod 2  │     │ Pod 3  │
         │ (v1.2) │     │ (v1.2) │     │ (v1.2) │
         └────────┘     └────────┘     └────────┘
```

## Stage 5: Monitoring and Observability

### What Happens
- Inference latency and throughput are monitored
- Prediction distributions are tracked
- Model performance degradation is detected
- System health metrics are collected
- Alerts are triggered for anomalies

### Infrastructure Requirements

**Monitoring Stack:**
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **ELK Stack** (Elasticsearch, Logstash, Kibana) - Log aggregation
- **Jaeger/Zipkin** - Distributed tracing

**Key Metrics to Monitor:**

**System Metrics:**
- Request rate (requests/second)
- Latency (p50, p95, p99)
- Error rate
- CPU/GPU utilization
- Memory usage

**Model Metrics:**
- Prediction distribution
- Confidence scores
- Feature distributions
- Data drift
- Concept drift

**Business Metrics:**
- Conversion rate
- Revenue per prediction
- User satisfaction

### Example Monitoring Setup

```python
from prometheus_client import Counter, Histogram

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'outcome']
)

inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration'
)

# Track metrics
with inference_duration.time():
    prediction = model.predict(input_data)

prediction_counter.labels(
    model_version='v1.2',
    outcome='success'
).inc()
```

## Stage 6: Model Retraining and Updates

### What Happens
- Performance degradation is detected
- New data is collected
- Model is retrained with updated data
- Retrained model is evaluated
- New model is deployed through CI/CD

### Infrastructure Requirements

**Automation:**
- **Scheduled retraining** - Periodic model updates
- **Triggered retraining** - When drift detected or new data available
- **CI/CD pipelines** - Automated testing and deployment

**Continuous Training:**
```
Data → Validate → Train → Evaluate → Deploy
  ↑                                     │
  └─────────────────────────────────────┘
         (Continuous Loop)
```

## Infrastructure Challenges in ML

### 1. Data Volume and Velocity
**Challenge:** ML requires massive amounts of data that grows continuously

**Solutions:**
- Distributed storage (HDFS, S3)
- Stream processing (Kafka, Flink)
- Data compression and efficient formats (Parquet, ORC)

### 2. Compute Intensity
**Challenge:** Training large models requires significant compute resources

**Solutions:**
- GPU clusters
- Distributed training (data parallelism, model parallelism)
- Cloud elasticity (scale up/down as needed)
- Spot instances for cost optimization

### 3. Model Versioning
**Challenge:** Need to track models, data, and code together

**Solutions:**
- Model registries (MLflow, DVC)
- Git for code
- Data versioning tools
- Reproducible experiments

### 4. Serving Latency
**Challenge:** Real-time predictions must be fast (<100ms)

**Solutions:**
- Model optimization (quantization, pruning)
- Caching
- Batch inference
- GPU acceleration
- Edge deployment

### 5. Monitoring Complexity
**Challenge:** Need to monitor both system and model performance

**Solutions:**
- Comprehensive observability stack
- Custom metrics for ML
- Automated alerting
- Drift detection

## Infrastructure Anti-Patterns to Avoid

### 1. Manual Deployments
**Problem:** Error-prone, slow, not reproducible

**Solution:** Automate with CI/CD pipelines

### 2. No Model Versioning
**Problem:** Can't roll back, can't reproduce results

**Solution:** Use model registry, version everything

### 3. Training in Production
**Problem:** Unstable, resource competition, security risks

**Solution:** Separate training and serving environments

### 4. Ignoring Data Quality
**Problem:** Garbage in, garbage out - models fail silently

**Solution:** Data validation, quality checks, monitoring

### 5. No Monitoring
**Problem:** Don't know when models degrade

**Solution:** Comprehensive monitoring and alerting

## The Role of ML Infrastructure Engineers

At each lifecycle stage, ML infrastructure engineers:

**Data Stage:**
- Build and maintain data pipelines
- Implement data versioning
- Ensure data quality

**Training Stage:**
- Provision GPU clusters
- Set up experiment tracking
- Optimize training performance

**Evaluation Stage:**
- Create evaluation frameworks
- Implement bias detection
- Build validation pipelines

**Deployment Stage:**
- Build serving infrastructure
- Implement deployment strategies
- Manage container orchestration

**Monitoring Stage:**
- Set up observability stack
- Define alerts and SLOs
- Build dashboards

**Retraining Stage:**
- Automate retraining pipelines
- Implement CI/CD for ML
- Manage model lifecycle

## Real-World Example: Recommendation System

Let's trace a recommendation system through the ML lifecycle:

**1. Data Collection:**
- User interactions logged to Kafka
- Batch job processes logs daily
- Features stored in feature store

**2. Training:**
- Nightly training on GPU cluster
- Experiment tracking with MLflow
- Best model registered

**3. Evaluation:**
- Offline metrics (AUC, NDCG)
- A/B test framework ready

**4. Deployment:**
- Canary deployment (5% traffic)
- Monitor metrics for 24 hours
- Full rollout if successful

**5. Monitoring:**
- Latency dashboard (target: <50ms)
- Click-through rate tracking
- Model drift detection

**6. Retraining:**
- Weekly retraining with new data
- Automated evaluation pipeline
- Auto-deploy if metrics improve

## Key Takeaways

1. **ML lifecycle is continuous** - Not a one-time process
2. **Infrastructure enables each stage** - Different requirements at each stage
3. **Automation is critical** - Manual processes don't scale
4. **Monitoring is essential** - Models can fail silently
5. **Everything should be versioned** - Data, code, models, config
6. **Production is different from development** - Need robust infrastructure

## Practical Exercise

Think about a simple ML use case (e.g., spam detection):

1. **List data requirements** - What data needed? Where does it come from?
2. **Identify compute needs** - What resources for training? For serving?
3. **Define success metrics** - How do you measure if the model works?
4. **Plan monitoring** - What would you track in production?
5. **Consider failure modes** - What could go wrong at each stage?

Write your answers and compare with the patterns discussed in this lesson.

## Self-Check Questions

1. What are the six stages of the ML lifecycle?
2. Why is model versioning important?
3. What's the difference between system metrics and model metrics?
4. What are three deployment strategies for ML models?
5. Why is monitoring critical for production ML systems?
6. What infrastructure is needed for the training stage?
7. What is data drift and why does it matter?

## Additional Resources

- [Google's MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Microsoft's MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)
- [Uber's Michelangelo Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- ["Machine Learning Engineering" by Andriy Burkov](http://www.mlebook.com/)

---

**Next Lesson:** [04-ml-frameworks.md](./04-ml-frameworks.md) - Working with PyTorch and TensorFlow for infrastructure engineers
