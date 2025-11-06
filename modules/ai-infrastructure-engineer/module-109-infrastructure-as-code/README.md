# Module 09: Infrastructure as Code (IaC)

## Module Overview

Infrastructure as Code (IaC) is a fundamental practice in modern DevOps and AI infrastructure management. This module teaches you how to define, deploy, and manage infrastructure using code, enabling version control, automation, reproducibility, and collaboration for AI/ML workloads.

## Why This Matters for AI Infrastructure

Managing AI infrastructure manually is error-prone, slow, and doesn't scale. IaC enables you to:

- **Reproducibility**: Deploy identical environments for dev, staging, and production
- **Version Control**: Track infrastructure changes in Git alongside application code
- **Automation**: Integrate infrastructure provisioning into CI/CD pipelines
- **Collaboration**: Enable teams to review and approve infrastructure changes
- **Disaster Recovery**: Rebuild entire infrastructure from code in minutes
- **Cost Management**: Easily spin up/down expensive GPU clusters
- **Multi-Cloud**: Deploy AI workloads across AWS, GCP, Azure with common patterns

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand IaC principles** and benefits for AI infrastructure
2. **Use Terraform** to provision cloud resources for ML workloads
3. **Manage infrastructure state** safely and collaborate with teams
4. **Apply Pulumi** for Python-based infrastructure management
5. **Implement GitOps workflows** for infrastructure changes
6. **Design modular, reusable infrastructure** components
7. **Secure sensitive data** in IaC (credentials, API keys)
8. **Implement CI/CD pipelines** for infrastructure deployment
9. **Apply best practices** for IaC at scale

## Prerequisites

- **Module 02**: Cloud Computing Fundamentals (AWS/GCP/Azure basics)
- **Module 03**: Containerization with Docker
- **Module 04**: Kubernetes fundamentals
- Basic understanding of cloud services (VMs, storage, networking)
- Familiarity with command-line tools
- Git basics

## Module Structure

This module contains **8 comprehensive lessons**:

### Lesson 01: Introduction to Infrastructure as Code
- What is IaC and why it matters
- IaC vs manual infrastructure management
- Declarative vs imperative approaches
- IaC tools landscape (Terraform, Pulumi, CloudFormation, Ansible)
- Infrastructure for AI/ML: special considerations

### Lesson 02: Terraform Fundamentals
- Installing and configuring Terraform
- HCL (HashiCorp Configuration Language) syntax
- Providers, resources, data sources
- Variables, outputs, locals
- Terraform CLI commands (init, plan, apply, destroy)
- Your first Terraform project

### Lesson 03: Terraform State Management
- Understanding Terraform state
- Local vs remote state backends
- State locking and collaboration
- S3 + DynamoDB backend setup
- State inspection and manipulation
- Workspaces for environment management

### Lesson 04: Building AI Infrastructure with Terraform
- Provisioning GPU instances (AWS p3, GCP A100)
- Setting up Kubernetes clusters (EKS, GKE, AKS)
- Creating storage for datasets (S3, GCS, Azure Blob)
- Networking and security groups
- IAM roles and permissions
- Complete ML platform example

### Lesson 05: Pulumi - Infrastructure as Software
- Introduction to Pulumi
- Python-based infrastructure code
- Pulumi vs Terraform comparison
- Deploying ML infrastructure with Pulumi
- Using loops, conditionals, and Python libraries
- Pulumi stacks and configuration

### Lesson 06: Advanced IaC Patterns
- Modules and code reusability
- Multi-environment strategies (dev/staging/prod)
- Multi-cloud deployments
- Dynamic configuration with external data
- Conditional resources
- For_each and count patterns

### Lesson 07: GitOps and IaC CI/CD
- GitOps principles for infrastructure
- Pull request workflows for infrastructure changes
- Automated testing for IaC (terraform validate, plan)
- CI/CD pipelines with GitHub Actions
- Atlantis for Terraform automation
- Policy as Code (Sentinel, OPA)

### Lesson 08: Security and Best Practices
- Managing secrets (AWS Secrets Manager, HashiCorp Vault)
- Preventing credential leaks
- Infrastructure security scanning (tfsec, Checkov)
- Cost estimation and optimization
- Tagging strategies
- Documentation and team collaboration
- Disaster recovery procedures

## Hands-On Exercises

This module includes practical exercises in the `exercises/` directory:

1. **Exercise 01**: Deploy a simple EC2 instance with Terraform
2. **Exercise 02**: Create an S3 bucket with versioning for ML datasets
3. **Exercise 03**: Provision a GKE cluster for ML workloads
4. **Exercise 04**: Build reusable Terraform modules
5. **Exercise 05**: Set up remote state with S3 backend
6. **Exercise 06**: Deploy ML infrastructure with Pulumi (Python)
7. **Exercise 07**: Create a GitOps workflow with GitHub Actions
8. **Exercise 08**: Implement secrets management with AWS Secrets Manager

**Estimated time per exercise**: 1-2 hours
**Total hands-on time**: 8-16 hours

## Assessment

- **Quiz**: 25 questions covering all lessons
- **Practical Exam**: Design and deploy a complete ML infrastructure
  - GPU instance for training
  - Kubernetes cluster for inference
  - S3 buckets for data/models
  - Monitoring and logging setup
  - All managed via Terraform with remote state

## Tools and Technologies

### Primary Tools
- **Terraform** (v1.5+) - HashiCorp's IaC tool
- **Pulumi** (Python SDK) - Modern IaC with real programming languages
- **AWS CLI** / **gcloud** / **az** - Cloud provider CLIs

### Supporting Tools
- **tfenv** - Terraform version manager
- **tflint** - Terraform linter
- **tfsec** - Security scanner for Terraform
- **Checkov** - Policy-as-code scanner
- **terraform-docs** - Generate documentation from Terraform modules
- **Atlantis** - Terraform pull request automation

### Cloud Providers
- AWS (primary examples)
- Google Cloud Platform
- Microsoft Azure

## Estimated Time

- **Reading and understanding**: 18-22 hours
- **Hands-on exercises**: 12-16 hours
- **Quiz and assessment**: 2-3 hours
- **Total**: 32-41 hours

## Advanced Exercises

After completing the foundational lessons, tackle these production IaC projects:

### Exercise 01: Production ML Infrastructure with Terraform (32-40 hours)
Build complete ML infrastructure using Terraform with best practices.

**Location:** `exercises/exercise-01-terraform-ml-infrastructure/`
**Prerequisites:** Completed all lessons, Terraform knowledge, cloud platform access
**Skills:** Infrastructure as Code, Terraform modules, state management, multi-environment

**Topics:**
- VPC and networking setup
- EKS/GKE cluster provisioning
- GPU node groups configuration
- S3/GCS storage and IAM policies
- RDS/Cloud SQL databases
- Monitoring stack (Prometheus, Grafana)
- Secrets management (Vault, AWS Secrets Manager)
- Multi-environment management (dev, staging, prod)
- Remote state and locking
- CI/CD for infrastructure changes

### Exercise 02: Multi-Cloud ML Infrastructure with Pulumi (32-40 hours)
Deploy ML infrastructure across multiple cloud providers using Pulumi.

**Location:** `exercises/exercise-02-pulumi-multicloud-ml/`
**Prerequisites:** Completed Exercise 01, programming experience (Python/TypeScript)
**Skills:** Multi-cloud deployment, Pulumi, cloud abstraction, disaster recovery

**Topics:**
- Pulumi programming model (Python/TypeScript)
- AWS + GCP + Azure deployment
- Cloud-agnostic abstractions
- Cost optimization strategies
- Disaster recovery and failover
- Cloud-specific optimizations
- Resource tagging and organization
- Policy as code (Pulumi CrossGuard)
- Stack management and collaboration

## Resources

See `resources.md` for:
- Official Terraform and Pulumi documentation
- Best practices guides
- Terraform Registry modules
- Video tutorials and courses
- Community resources
- Certification information

## Learning Path

**This module builds on:**
- Module 02: Cloud Computing Fundamentals
- Module 03: Containerization
- Module 04: Kubernetes

**This module prepares you for:**
- Module 10: LLM Infrastructure (deploying LLM infrastructure with IaC)
- Project 02: MLOps Pipeline (infrastructure automation)
- Project 03: LLM Deployment (scalable LLM infrastructure)

## Real-World Applications

After completing this module, you'll be able to:

- Provision complete ML training environments on-demand
- Deploy Kubernetes clusters for model serving
- Manage multi-environment ML infrastructure (dev/staging/prod)
- Implement disaster recovery for critical ML systems
- Collaborate with teams on infrastructure changes
- Reduce infrastructure costs through automation
- Meet compliance requirements with auditable infrastructure

## Getting Started

Begin with **Lesson 01: Introduction to Infrastructure as Code** to understand the fundamentals, then progress through the lessons sequentially.

---

**Module Difficulty**: Intermediate
**Recommended Background**: Cloud computing basics, command-line experience
**Output**: Production-ready IaC skills for AI infrastructure

Let's start building infrastructure as code! 🚀
