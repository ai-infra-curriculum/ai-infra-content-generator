# Lesson 03: ML Lifecycle and Infrastructure Requirements

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the complete ML lifecycle from data to deployment
- Identify infrastructure requirements for each stage
- Recognize the role of MLOps in the ML lifecycle
- Plan infrastructure needs for ML projects

## Prerequisites

- Completed Lessons 01-02
- Basic understanding of machine learning concepts
- Familiarity with data processing pipelines

## Lesson Duration

**Estimated Time**: 2-3 hours

---

## 1. The ML Lifecycle: Overview

### 1.1 Stages of ML Development

The machine learning lifecycle consists of several interconnected stages:

```
Data Collection → Data Processing → Feature Engineering
                                            ↓
                                      Model Training
                                            ↓
                                    Model Evaluation
                                            ↓
                                    Model Deployment
                                            ↓
                                      Monitoring → (feedback loop)
```

### 1.2 Traditional Software vs ML Systems

| Aspect | Traditional Software | ML Systems |
|--------|---------------------|------------|
| **Code Changes** | Frequent | Less frequent |
| **Data Changes** | Rare | Constant |
| **Testing** | Deterministic | Statistical |
| **Deployment** | Binary versioning | Model + data versioning |
| **Monitoring** | System metrics | Model performance + system metrics |
| **Debugging** | Stack traces | Data drift, model decay |

---

## 2. Infrastructure Requirements by Stage

### 2.1 Data Collection and Storage

**Infrastructure Needs:**
- **Object Storage**: S3, GCS, Azure Blob (for raw data, datasets)
- **Data Lakes**: Centralized storage for structured/unstructured data
- **Databases**: PostgreSQL, MongoDB (for metadata, labels)
- **Streaming**: Kafka, Kinesis (for real-time data ingestion)

**Key Considerations:**
- Storage costs (optimize for infrequent access)
- Data versioning and lineage
- Access controls and security
- Scalability for growing datasets

**Example Architecture:**
```
Data Sources → Kafka → Data Lake (S3) → Processing Pipeline
                                              ↓
                                        Feature Store
```

### 2.2 Data Processing and Feature Engineering

**Infrastructure Needs:**
- **Compute**: CPUs for data processing (batch or stream)
- **Frameworks**: Apache Spark, Dask, Ray for distributed processing
- **Workflow Orchestration**: Airflow, Prefect, Dagster
- **Feature Stores**: Feast, Tecton (for feature management)

**Key Considerations:**
- Processing large datasets efficiently
- Scheduling and dependency management
- Data quality validation
- Feature reusability across models

**Common Tools:**
- **Apache Spark**: Distributed data processing
- **Apache Airflow**: Workflow orchestration
- **DVC**: Data versioning
- **Great Expectations**: Data validation

### 2.3 Model Training

**Infrastructure Needs:**
- **GPUs**: NVIDIA GPUs for deep learning (A100, V100, T4)
- **CPUs**: For traditional ML (scikit-learn, XGBoost)
- **Memory**: Large RAM for in-memory processing
- **Storage**: Fast SSD/NVMe for training data access
- **Experiment Tracking**: MLflow, Weights & Biases

**Key Considerations:**
- GPU vs CPU cost-performance trade-offs
- Distributed training for large models
- Experiment reproducibility
- Hyperparameter tuning at scale

**Training Infrastructure Options:**
```
Local Development → GPU VM (single node) → GPU Cluster (multi-node)
     ↓                     ↓                          ↓
  Prototyping        Medium models            Large models/distributed
  Free/$50/mo          $200-500/mo              $1000+/mo
```

### 2.4 Model Evaluation and Validation

**Infrastructure Needs:**
- **Compute**: For running evaluation scripts
- **Storage**: For evaluation datasets and results
- **Visualization**: TensorBoard, MLflow UI, custom dashboards

**Key Considerations:**
- Holdout test sets (never touched during training)
- Multiple evaluation metrics
- Model comparison frameworks
- A/B testing infrastructure (for production comparison)

### 2.5 Model Deployment and Serving

**Infrastructure Needs:**
- **Serving Frameworks**: TorchServe, TensorFlow Serving, vLLM
- **API Layer**: FastAPI, Flask, gRPC
- **Containerization**: Docker, Kubernetes
- **Load Balancing**: NGINX, cloud load balancers
- **Auto-scaling**: Kubernetes HPA, cloud auto-scaling

**Key Considerations:**
- Latency requirements (real-time vs batch)
- Throughput (requests per second)
- Model size and memory requirements
- Cold start optimization

**Deployment Patterns:**
```
Batch Inference: Scheduled jobs, process large datasets
Real-time Inference: REST API, <100ms latency
Streaming Inference: Process events as they arrive
```

### 2.6 Monitoring and Maintenance

**Infrastructure Needs:**
- **Metrics Collection**: Prometheus, CloudWatch, Datadog
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Dashboards**: Grafana, custom dashboards
- **Alerting**: PagerDuty, Slack integrations
- **Model Monitoring**: Evidently AI, Arize, WhyLabs

**Key Monitoring Dimensions:**
1. **System Metrics**: CPU, memory, GPU utilization, latency
2. **Model Metrics**: Prediction accuracy, confidence scores
3. **Data Metrics**: Input distribution, feature drift
4. **Business Metrics**: User engagement, revenue impact

**Common Issues to Monitor:**
- **Model Drift**: Model performance degrades over time
- **Data Drift**: Input distribution changes
- **Concept Drift**: Relationship between features and target changes
- **System Issues**: High latency, OOM errors, API failures

---

## 3. MLOps: Bridging ML and Operations

### 3.1 What is MLOps?

**MLOps** (Machine Learning Operations) is the practice of collaboration between data scientists and operations teams to:
- Automate ML workflows
- Ensure reproducibility
- Monitor model performance
- Enable continuous deployment

**MLOps = ML + DevOps + Data Engineering**

### 3.2 Key MLOps Principles

1. **Version Everything**: Code, data, models, configurations
2. **Automate Pipelines**: CI/CD for ML (training, testing, deployment)
3. **Monitor Continuously**: Track model and system performance
4. **Enable Reproducibility**: Anyone should be able to reproduce results
5. **Collaborate Effectively**: Tools for data scientists and engineers

### 3.3 MLOps Maturity Levels

**Level 0: Manual**
- Manual data analysis and model training
- Manual deployment
- No automation or monitoring

**Level 1: Automated Training**
- Automated training pipelines
- Experiment tracking
- Basic model versioning

**Level 2: Automated Deployment**
- CI/CD for models
- Automated testing and validation
- Monitoring and alerting

**Level 3: Full MLOps**
- End-to-end automation
- Feature stores
- Continuous training and deployment
- Advanced monitoring and feedback loops

---

## 4. Infrastructure Cost Considerations

### 4.1 Cost Components

| Component | Typical Cost | Optimization Strategies |
|-----------|-------------|------------------------|
| **Storage** | $0.02-0.10/GB/month | Use cheaper storage tiers, compress data |
| **Compute (CPU)** | $0.05-0.50/hour | Use spot instances, right-size VMs |
| **GPU** | $0.50-30/hour | Use spot instances, optimize utilization |
| **Networking** | $0.01-0.12/GB | Minimize cross-region transfers |
| **Managed Services** | Variable | Compare with self-hosted options |

### 4.2 Cost Optimization Tips

1. **Use Spot/Preemptible Instances**: 60-90% cost savings for training
2. **Auto-scaling**: Scale down when not in use
3. **Data Lifecycle Policies**: Move old data to cheaper storage
4. **Right-sizing**: Match compute to workload needs
5. **Reserved Instances**: For predictable workloads (1-3 year commitment)
6. **GPU Sharing**: Run multiple models on same GPU (MIG, MPS)

### 4.3 Sample Budget for Small ML Project

```
Monthly Infrastructure Costs:
- Storage (100GB): $10
- Training GPU (10 hours/month): $100
- Inference (1 CPU instance, 24/7): $30
- Monitoring/Logging: $20
- Data Transfer: $10
--------------------------------
Total: ~$170/month
```

---

## 5. Hands-On Exercise: Plan Infrastructure for a Project

### Exercise: Image Classification Service

**Scenario**: You need to build an image classification service that:
- Processes 1 million images per day
- Requires <200ms latency per request
- Handles 50 requests per second at peak
- Trains monthly with new data

**Your Task**: Design the infrastructure architecture

**Consider:**
1. Data storage needs (raw images, processed features, models)
2. Training infrastructure (GPU type, duration, frequency)
3. Serving infrastructure (number of instances, auto-scaling)
4. Monitoring and logging requirements
5. Estimated monthly costs

**Template:**
```markdown
## Infrastructure Plan

### 1. Data Storage
- Raw images: [your answer]
- Processed data: [your answer]
- Models: [your answer]

### 2. Training Infrastructure
- GPU type: [your answer]
- Training frequency: [your answer]
- Estimated cost: [your answer]

### 3. Serving Infrastructure
- Compute instances: [your answer]
- Auto-scaling config: [your answer]
- Estimated cost: [your answer]

### 4. Monitoring
- Tools: [your answer]
- Metrics tracked: [your answer]

### 5. Total Estimated Cost
[your answer]
```

**Solution Discussion**: Compare your solution with peers or mentors. There's no single right answer!

---

## 6. Key Takeaways

1. ✅ ML lifecycle has distinct stages, each with unique infrastructure needs
2. ✅ Infrastructure must support data storage, processing, training, serving, and monitoring
3. ✅ MLOps practices bridge the gap between ML development and production deployment
4. ✅ Cost optimization is critical—use spot instances, auto-scaling, and right-sizing
5. ✅ Monitoring is essential for maintaining model performance in production

---

## 7. Additional Resources

### Reading
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (Google Cloud)
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (Research Paper)
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Google)

### Tools to Explore
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **Airflow**: Workflow orchestration
- **Prometheus + Grafana**: Monitoring and dashboards

### Videos
- [MLOps Explained](https://www.youtube.com/results?search_query=mlops+explained) (YouTube search)
- [Full Stack Deep Learning - MLOps](https://fullstackdeeplearning.com/)

---

## 8. Quiz

Test your understanding:

1. **What are the main stages of the ML lifecycle?**
   - [ ] A. Data, Training, Deployment
   - [ ] B. Data Collection, Processing, Training, Evaluation, Deployment, Monitoring
   - [ ] C. Training, Testing, Production
   - [ ] D. Development, Staging, Production

2. **Which infrastructure component is most important for model training?**
   - [ ] A. Load balancers
   - [ ] B. GPUs
   - [ ] C. Databases
   - [ ] D. Networking

3. **What is the primary benefit of MLOps?**
   - [ ] A. Faster model training
   - [ ] B. Better model accuracy
   - [ ] C. Automation and reproducibility
   - [ ] D. Lower data storage costs

4. **Which type of instance provides 60-90% cost savings for training?**
   - [ ] A. Reserved instances
   - [ ] B. On-demand instances
   - [ ] C. Spot/Preemptible instances
   - [ ] D. Dedicated hosts

5. **What should you monitor in production ML systems?**
   - [ ] A. Only system metrics (CPU, memory)
   - [ ] B. Only model accuracy
   - [ ] C. System metrics, model metrics, data metrics, and business metrics
   - [ ] D. Only business metrics

**Answers**: 1-B, 2-B, 3-C, 4-C, 5-C

---

## Next Steps

✅ Complete the hands-on exercise above
✅ Proceed to [Lesson 04: ML Frameworks Overview](./04-ml-frameworks-overview.md)
✅ Start thinking about Project 01: Basic Model Serving

---

**Questions?** Open an issue or ask in the GitHub Discussions!
