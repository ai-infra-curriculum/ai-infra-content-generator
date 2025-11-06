# Lesson 05: Cloud Platforms for ML - Introduction

## Learning Objectives

- Understand cloud computing fundamentals (IaaS, PaaS, SaaS)
- Compare major cloud providers (AWS, GCP, Azure)
- Set up cloud accounts and configure billing alerts
- Deploy first ML workload on cloud

## Duration: 3-4 hours

---

## 1. Cloud Computing Models

### IaaS, PaaS, SaaS

```
┌────────────────────────────────────────────┐
│ SaaS (Software as a Service)              │
│ Examples: Gmail, Salesforce, Google Docs  │
└────────────────────────────────────────────┘
┌────────────────────────────────────────────┐
│ PaaS (Platform as a Service)              │
│ Examples: Heroku, Google App Engine       │
└────────────────────────────────────────────┘
┌────────────────────────────────────────────┐
│ IaaS (Infrastructure as a Service)        │
│ Examples: EC2, Compute Engine, Azure VMs  │
└────────────────────────────────────────────┘
```

**For ML Infrastructure**: We primarily use **IaaS** and some **PaaS**

---

## 2. Major Cloud Providers

### 2.1 AWS (Amazon Web Services)

**Market Leader**: 32% market share (2024)

**Key ML Services:**
- **EC2**: Virtual machines (p4d, p5 for GPUs)
- **S3**: Object storage for data and models
- **SageMaker**: Managed ML platform
- **Lambda**: Serverless compute
- **EKS**: Managed Kubernetes

**Pros:**
- Largest ecosystem and service catalog
- Best documentation and community
- Most mature ML services
- Global infrastructure (33 regions)

**Cons:**
- Complex pricing
- Steeper learning curve
- Can be expensive without optimization

### 2.2 GCP (Google Cloud Platform)

**ML Innovator**: 11% market share (2024)

**Key ML Services:**
- **Compute Engine**: Virtual machines
- **Cloud Storage**: Object storage
- **Vertex AI**: Managed ML platform (best-in-class)
- **Cloud Functions**: Serverless
- **GKE**: Managed Kubernetes (best Kubernetes experience)
- **TPUs**: Google's custom AI accelerators

**Pros:**
- Best ML/AI services (Vertex AI, AutoML)
- Superior Kubernetes (GKE)
- TPUs for specific workloads
- Cleaner pricing model
- BigQuery for data analytics

**Cons:**
- Smaller ecosystem than AWS
- Fewer regions than AWS
- Less enterprise adoption

### 2.3 Azure (Microsoft Azure)

**Enterprise Focus**: 23% market share (2024)

**Key ML Services:**
- **Virtual Machines**: Compute instances
- **Blob Storage**: Object storage
- **Azure ML**: Managed ML platform
- **Functions**: Serverless
- **AKS**: Managed Kubernetes

**Pros:**
- Best for Microsoft-centric organizations
- Strong enterprise features and compliance
- Good Windows support
- Active Directory integration
- Growing ML capabilities

**Cons:**
- ML services less mature than AWS/GCP
- Documentation can be confusing
- Pricing complexity

---

## 3. Cloud Provider Comparison for ML

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| **GPU Instances** | Excellent (P4d, P5) | Excellent | Good |
| **TPUs** | ❌ | ✅ Best | ❌ |
| **Managed ML** | SageMaker (Good) | Vertex AI (Excellent) | Azure ML (Good) |
| **Kubernetes** | EKS (Good) | GKE (Excellent) | AKS (Good) |
| **Pricing** | Complex | Cleaner | Complex |
| **Free Tier** | 750h/month | $300 credit | $200 credit |
| **Learning Curve** | Medium-High | Medium | Medium-High |
| **ML Ecosystem** | Largest | Innovative | Growing |

**Recommendation for Beginners**: Start with **GCP** for best ML experience, or **AWS** for broadest skills transferability.

---

## 4. Hands-On: Set Up Your Cloud Account

### 4.1 Create AWS Account (Option 1)

**TODO: Follow these steps**

1. **Sign up**: Visit [aws.amazon.com](https://aws.amazon.com/)
2. **Provide credit card** (required, but free tier available)
3. **Set up billing alerts** (CRITICAL):
   ```
   CloudWatch → Billing → Create Alarm
   Alert when: EstimatedCharges > $10
   ```
4. **Create IAM user** (don't use root account):
   ```bash
   # Install AWS CLI
   pip install awscli

   # Configure credentials
   aws configure
   AWS Access Key ID: [your-key]
   AWS Secret Access Key: [your-secret]
   Default region: us-east-1
   ```

5. **Launch first EC2 instance**:
   - Go to EC2 Dashboard
   - Launch Instance → t2.micro (free tier)
   - Create key pair
   - SSH into instance:
     ```bash
     ssh -i "key.pem" ec2-user@[public-ip]
     ```

### 4.2 Create GCP Account (Option 2)

**TODO: Follow these steps**

1. **Sign up**: Visit [cloud.google.com](https://cloud.google.com/)
2. **$300 free credit** (90 days, no charges after)
3. **Create project**:
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   gcloud init

   # Create project
   gcloud projects create my-ml-project
   gcloud config set project my-ml-project
   ```

4. **Set up billing budget**:
   - Billing → Budgets & alerts
   - Set budget: $50/month
   - Alert at 50%, 90%, 100%

5. **Launch first Compute Engine instance**:
   ```bash
   gcloud compute instances create my-first-instance \
     --zone=us-central1-a \
     --machine-type=e2-micro \
     --image-family=debian-11 \
     --image-project=debian-cloud

   # SSH into instance
   gcloud compute ssh my-first-instance --zone=us-central1-a
   ```

---

## 5. Cost Management Best Practices

### 5.1 Always Set Billing Alerts

**CRITICAL**: Set up alerts BEFORE creating resources

```
AWS: CloudWatch Billing Alarms ($10, $50, $100)
GCP: Budget Alerts (50%, 90%, 100% of $50)
Azure: Cost Management + Billing Alerts
```

### 5.2 Use Free Tiers

**AWS Free Tier** (12 months):
- 750 hours/month t2.micro EC2
- 5GB S3 storage
- 1 million Lambda requests

**GCP Free Tier**:
- $300 credit (90 days)
- Always free: 1 e2-micro instance, 5GB storage

**Azure Free Tier**:
- $200 credit (30 days)
- Always free: Limited services

### 5.3 Stop Instances When Not Using

```bash
# AWS: Stop instance (not terminate!)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# GCP: Stop instance
gcloud compute instances stop my-instance

# Schedule automatic shutdowns (saves 60-80% costs)
```

---

## 6. Hands-On Exercise: Deploy ML Model on Cloud

### Task: Deploy a Simple ML API on Cloud

**TODO: Complete this exercise**

1. **Launch VM with GPU** (optional, use CPU for free tier)
2. **Install dependencies**:
   ```bash
   # SSH into instance
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install torch fastapi uvicorn
   ```

3. **Create simple ML API**:
   ```python
   # TODO: Create app.py with FastAPI endpoint
   # that serves a pre-trained model
   ```

4. **Run API**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Test from local machine**:
   ```bash
   curl http://[VM-PUBLIC-IP]:8000/predict
   ```

6. **Don't forget to STOP the instance!**

---

## 7. Key Takeaways

- ✅ Cloud platforms enable scalable ML infrastructure
- ✅ AWS (breadth), GCP (ML focus), Azure (enterprise)
- ✅ Always set billing alerts BEFORE spending money
- ✅ Use free tiers for learning and prototyping
- ✅ Stop/delete resources when not in use

---

## 8. Additional Resources

- [AWS Free Tier](https://aws.amazon.com/free/)
- [GCP Free Tier](https://cloud.google.com/free)
- [Azure Free Tier](https://azure.microsoft.com/en-us/free/)
- [Coursera: Cloud Architecture Specialization](https://www.coursera.org/)

---

## Next Steps

✅ Create cloud account and set up billing alerts
✅ Complete hands-on exercise
✅ Proceed to [Lesson 06: Model Serving Basics](./06-model-serving-basics.md)
