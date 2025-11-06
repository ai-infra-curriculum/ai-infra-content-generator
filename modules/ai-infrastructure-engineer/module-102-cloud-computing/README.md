# Module 02: Cloud Computing for ML Infrastructure

**Duration:** 50 hours
**Difficulty:** Intermediate
**Prerequisites:** Module 01 complete

## Module Overview

This module provides an in-depth exploration of cloud computing platforms for ML infrastructure. You'll learn to architect, deploy, and manage ML systems across AWS, GCP, and Azure, with a focus on cost optimization, scalability, and production best practices.

## Learning Objectives

By the end of this module, you will be able to:

1. **Compare and choose** appropriate cloud platforms for ML workloads
2. **Design cloud-native** ML architectures across AWS, GCP, and Azure
3. **Optimize costs** for ML infrastructure in the cloud
4. **Deploy and manage** compute resources (VMs, containers, serverless)
5. **Configure storage** solutions for datasets and models
6. **Implement networking** for secure and performant ML systems
7. **Use managed ML services** (SageMaker, Vertex AI, Azure ML)
8. **Build multi-cloud** and hybrid strategies

## Module Structure

### Lesson 01: Cloud Architecture for ML (6 hours)
- Cloud architecture patterns
- Compute, storage, networking fundamentals
- ML-specific considerations
- Cost modeling and estimation
- Hands-on: Design ML system architecture

### Lesson 02: AWS for ML Infrastructure (7 hours)
- AWS ecosystem overview
- EC2, S3, EBS, VPC
- EKS (Elastic Kubernetes Service)
- SageMaker deep dive
- IAM and security
- Hands-on: Deploy ML system on AWS

### Lesson 03: Google Cloud Platform for ML (7 hours)
- GCP ecosystem overview
- Compute Engine, Cloud Storage, GKE
- Vertex AI platform
- TPU access and usage
- GCP networking
- Hands-on: Deploy ML system on GCP

### Lesson 04: Azure for ML Infrastructure (6 hours)
- Azure ecosystem overview
- Virtual Machines, Blob Storage, AKS
- Azure Machine Learning
- Azure OpenAI integration
- Enterprise features
- Hands-on: Deploy ML system on Azure

### Lesson 05: Cloud Storage for ML (6 hours)
- Object storage (S3, GCS, Blob)
- Block storage (EBS, Persistent Disks)
- File storage (EFS, Filestore)
- Data lakes and data warehouses
- Caching strategies
- Hands-on: Optimize data pipeline

### Lesson 06: Cloud Networking for ML (6 hours)
- VPC and subnet design
- Load balancers and traffic management
- CDN for model serving
- VPN and hybrid connectivity
- Service mesh (Istio)
- Hands-on: Secure multi-tier ML system

### Lesson 07: Managed ML Services (6 hours)
- SageMaker vs Vertex AI vs Azure ML
- Training jobs and endpoints
- Feature stores
- Model registry and versioning
- AutoML capabilities
- Hands-on: Compare managed services

### Lesson 08: Multi-Cloud and Cost Optimization (6 hours)
- Multi-cloud strategies
- Cloud cost management
- Reserved instances and savings plans
- Spot instances for training
- Cost monitoring and alerts
- Hands-on: Optimize cloud spending

## Prerequisites

**Required:**
- Completed Module 01
- Basic understanding of cloud concepts
- Familiarity with command-line interfaces
- Active account on at least one cloud platform (free tier)

**Recommended:**
- Linux/Unix command-line experience
- Basic networking knowledge
- Understanding of infrastructure as code

## Required Tools and Accounts

### Cloud Platform Accounts
- **AWS**: Free tier account (https://aws.amazon.com/free/)
- **GCP**: $300 free credit (https://cloud.google.com/free)
- **Azure**: $200 free credit (https://azure.microsoft.com/free/)

### CLI Tools
```bash
# AWS CLI
pip install awscli

# GCP SDK
# Download from: https://cloud.google.com/sdk/docs/install

# Azure CLI
pip install azure-cli

# Terraform (Infrastructure as Code)
# Download from: https://www.terraform.io/downloads
```

### Optional Tools
- Pulumi (Alternative IaC)
- Cloud cost monitoring tools
- Network debugging tools (traceroute, curl, etc.)

## Assessment

### Quizzes
- Mid-module quiz (after Lesson 04): 20 questions
- Final quiz (after Lesson 08): 30 questions

### Practical Exercises
- Exercise 01: Multi-cloud deployment
- Exercise 02: Cost optimization challenge
- Exercise 03: Network architecture design

### Capstone Project
**Project:** Deploy a production-ready ML system with:
- Multi-region deployment
- Auto-scaling
- Cost under $50/month
- 99.9% availability
- Comprehensive monitoring

## Learning Path

```
Module 01 (Complete) ──► Module 02 (Current) ──► Module 03
    ↓                         ↓                      ↓
Foundations            Cloud Computing        Kubernetes
```

## Time Commitment

- **Lessons**: 50 hours (6-7 hours per lesson)
- **Exercises**: 8-10 hours
- **Quizzes**: 2-3 hours
- **Capstone Project**: 15-20 hours
- **Total**: ~75-85 hours

**Recommended pace:**
- Part-time (10h/week): 7-8 weeks
- Full-time (40h/week): 2 weeks

## Cloud Credits and Free Tiers

### AWS Free Tier
- **750 hours/month** of t2.micro EC2 (12 months)
- **5 GB** of S3 storage
- **15 GB** of bandwidth out
- Many other services with free tier

### GCP Free Tier
- **$300 credit** for 90 days
- **e2-micro instance** free (always free tier)
- **30 GB** of storage
- **1 GB** of egress to North America

### Azure Free Tier
- **$200 credit** for 30 days
- **750 hours/month** of B1S VM (12 months)
- **5 GB** of Blob storage
- Many other services

**Important:** Set up billing alerts to avoid unexpected charges!

## Success Criteria

You've mastered this module when you can:

- [ ] Architect ML systems for specific cloud platforms
- [ ] Deploy and manage ML workloads on AWS, GCP, and Azure
- [ ] Optimize cloud costs for ML infrastructure
- [ ] Configure secure networking for ML systems
- [ ] Choose appropriate storage solutions
- [ ] Use managed ML services effectively
- [ ] Implement multi-cloud strategies
- [ ] Monitor and troubleshoot cloud deployments

## Resources

### Documentation
- [AWS Documentation](https://docs.aws.amazon.com/)
- [GCP Documentation](https://cloud.google.com/docs)
- [Azure Documentation](https://docs.microsoft.com/azure/)

### Books
- "Architecting for the Cloud: AWS Best Practices"
- "Google Cloud Platform in Action"
- "Azure for Architects"

### Online Resources
- AWS Training and Certification
- Google Cloud Skills Boost
- Microsoft Learn
- A Cloud Guru
- Linux Academy

### Community
- AWS Community Builders
- GCP Community
- Azure Tech Community
- Reddit: r/aws, r/googlecloud, r/azure

## Next Steps

After completing this module:
1. Complete Module 02 quiz with ≥80% score
2. Finish capstone project
3. Proceed to Module 03: Kubernetes Deep Dive
4. Or explore Module 04: Monitoring and Observability

---

**Ready to start?** Begin with [Lesson 01: Cloud Architecture for ML](./01-cloud-architecture.md)

**Need help?** Review [Module 01](../01-foundations/) if cloud concepts are unclear.

**Cost concerns?** All lessons can be completed within free tier limits!
