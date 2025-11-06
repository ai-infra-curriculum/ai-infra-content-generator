# AI Infrastructure Architect - Complete Curriculum Index

**Repository**: ai-infra-architect-solutions
**Level**: AI Infrastructure Architect (Level 3)
**Target Role**: L6/L7 at Big Tech, Director of ML Infrastructure, Principal Engineer
**Salary Range**: $200K-$300K base, $350K-$600K total comp (2025, US)
**Total Hours**: 425 hours
**Last Updated**: October 25, 2025

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Complete Project Catalog](#complete-project-catalog)
3. [Skills Matrix](#skills-matrix)
4. [Learning Path Recommendations](#learning-path-recommendations)
5. [Project Dependencies](#project-dependencies)
6. [Technology Coverage](#technology-coverage)
7. [Artifact Types Catalog](#artifact-types-catalog)
8. [Career Alignment](#career-alignment)

---

## Repository Overview

### Scope and Focus

This repository is fundamentally different from Engineer-level repositories:

**Artifact Distribution**:
- **60% Architecture Artifacts**: Business cases, ADRs, C4 diagrams, governance frameworks, stakeholder presentations
- **40% Reference Implementations**: Code that validates architectural decisions (not production-ready)

**Primary Deliverables**:
- Complete C4 architecture diagrams (Context, Container, Component, Deployment)
- 10+ Architecture Decision Records (ADRs) per project
- Comprehensive business cases with NPV/ROI/TCO analysis
- Stakeholder presentations (executive, technical, operational)
- Governance frameworks and compliance documentation
- Reference implementations in Terraform and Kubernetes

**Unique Characteristics**:
- Every project tied to measurable business value ($5M-$30M+ impact)
- Architecture optimized for 99.95%+ uptime at enterprise scale
- Compliance and governance as first-class concerns (not afterthoughts)
- Multi-stakeholder communication (C-suite, engineers, auditors, legal)

### Key Differentiators

| Aspect | Engineer Repos | This Architect Repo |
|--------|---------------|---------------------|
| Primary Focus | Working code | Architecture artifacts |
| Documentation | How-to guides, API docs | Business cases, ADRs, financial models |
| Scope | Single system/service | Enterprise platforms, multi-year strategies |
| Audience | Technical teams | C-suite, architects, technical leads |
| Success Metrics | System performance | Business value, ROI, strategic alignment |
| Decisions | Implementation details | Strategic architecture trade-offs |
| Timeline | Weeks to months | Months to years (2-5 year strategies) |

---

## Complete Project Catalog

### Project 301: Enterprise MLOps Platform ⭐ (FLAGSHIP)

**Duration**: 80 hours
**Complexity**: High (Enterprise Scale)
**Business Impact**: $30M NPV over 3 years

**Executive Summary**:
Design a comprehensive MLOps platform supporting 100+ data scientists across 20+ teams with full model lifecycle management, governance, and compliance.

**Business Value**:
- **$30M NPV** ($15M investment → $45M value creation)
- **35% cost reduction** in ML infrastructure spend
- **60% faster model deployment** (6 weeks → 2.5 weeks)
- **10x improvement** in model governance and auditability
- **Compliance ready**: SOC2, HIPAA, GDPR

**Learning Outcomes**:
- Design enterprise-scale MLOps platforms
- Build comprehensive business cases with multi-year ROI
- Create complete C4 diagram sets (4 levels)
- Write ADRs for strategic technology decisions
- Develop model governance and compliance frameworks
- Present to C-suite and board of directors

**Key Technologies**:
- Kubernetes (EKS), MLflow, Feast, KServe
- Kubeflow Pipelines, Airflow
- PostgreSQL, S3, Prometheus/Grafana
- Terraform, Helm

**Architecture Artifacts**:
- README.md (8,100+ lines) with complete business case
- ARCHITECTURE.md (10,000+ words)
- 10+ ADRs (technology stack, feature store, multi-tenancy, governance)
- Complete C4 diagram set
- Business case with NPV/ROI/sensitivity analysis
- Stakeholder presentations (executive, technical, operational)
- Governance framework (model approval, audit procedures)
- Reference Terraform + Kubernetes implementation

**Prerequisites**:
- Senior Engineer track completion
- Experience with Kubernetes and MLOps tools
- Basic understanding of ROI and business metrics

**Best For**:
- Those transitioning from Senior Engineer to Architect
- Platform architects designing MLOps infrastructure
- Technical leaders presenting to executives

---

### Project 302: Multi-Cloud AI Infrastructure

**Duration**: 100 hours
**Complexity**: Very High (Global Scale)
**Business Impact**: $8M annual cost savings

**Executive Summary**:
Architect a multi-cloud AI infrastructure spanning AWS, GCP, and Azure with data sovereignty compliance, HA/DR (RTO<1hr, RPO<15min), and cost optimization across clouds.

**Business Value**:
- **$8M annual cost savings** through cloud optimization
- **99.95% uptime** with disaster recovery
- **Regulatory compliance** across 15 countries (GDPR, CCPA, regional laws)
- **Cloud vendor independence** (risk mitigation)
- **Best-of-breed** services across clouds

**Learning Outcomes**:
- Design multi-cloud architectures with true portability
- Implement HA/DR strategies (active-active, active-passive)
- Navigate data sovereignty and compliance requirements
- Optimize costs across multiple cloud providers
- Create cloud vendor selection frameworks
- Migrate from single to multi-cloud (18-month phased plan)

**Key Technologies**:
- Terraform (multi-cloud IaC)
- Crossplane (cloud-agnostic control plane)
- Kubernetes Federation (multi-cluster)
- AWS (EKS, S3, Bedrock), GCP (GKE, Vertex AI), Azure (AKS, Azure ML)
- Inter-cloud networking (VPN, Direct Connect, ExpressRoute)

**Architecture Artifacts**:
- Multi-cloud vendor selection framework
- Architecture for 3 clouds with detailed comparison
- HA/DR plan with RTO/RPO analysis and runbooks
- Data sovereignty compliance framework
- FinOps cost optimization strategy
- Migration roadmap (18 months, 4 phases)
- 10+ ADRs (cloud strategy, data residency, DR approach, cost optimization)

**Prerequisites**:
- Project 301 completion (or equivalent MLOps experience)
- Experience with at least one cloud provider (AWS/GCP/Azure)
- Understanding of networking and security fundamentals

**Best For**:
- Global organizations with data residency requirements
- Architects avoiding vendor lock-in
- Those designing cloud migration strategies
- FinOps and cost optimization focus

---

### Project 303: LLM Platform with RAG

**Duration**: 90 hours
**Complexity**: Very High (Cutting Edge)
**Business Impact**: $4.2M annual savings

**Executive Summary**:
Design an enterprise LLM platform with RAG capabilities serving 10,000+ users, featuring responsible AI governance, cost optimization (70% reduction), and safety guardrails.

**Business Value**:
- **$4.2M annual cost savings** ($500K → $150K/month)
- **10x throughput improvement** via vLLM and TensorRT-LLM
- **Enterprise compliance** (data privacy, content safety, audit trails)
- **Sub-second latency** at scale (10K+ concurrent users)
- **Responsible AI** governance and bias mitigation

**Learning Outcomes**:
- Design LLM infrastructure at enterprise scale
- Architect RAG (Retrieval-Augmented Generation) systems
- Implement LLM safety and governance frameworks
- Optimize LLM inference costs (self-hosting vs API)
- Create responsible AI compliance frameworks
- Build model selection and evaluation criteria

**Key Technologies**:
- vLLM, TensorRT-LLM (inference optimization)
- Vector databases (Pinecone, Weaviate, Qdrant)
- LangChain, LlamaIndex (LLM orchestration)
- GPU clusters (A100, H100)
- Model routing and load balancing

**Architecture Artifacts**:
- LLM infrastructure architecture (GPU clusters, inference optimization)
- Model selection framework (20+ LLMs evaluated)
- Complete RAG system design with vector database architecture
- LLM safety and governance framework (bias, toxicity, hallucination mitigation)
- Cost-performance optimization analysis (70% cost reduction)
- Reference vLLM/TensorRT-LLM implementation
- 10+ ADRs (LLM selection, RAG architecture, inference engine, safety guardrails)

**Prerequisites**:
- Project 301 completion (MLOps platform foundation)
- Understanding of LLMs and transformers
- GPU computing basics

**Best For**:
- Organizations deploying GenAI/LLM applications
- Architects focused on AI safety and responsible AI
- Cost-conscious LLM infrastructure design
- Those building internal ChatGPT-like platforms

---

### Project 304: Data Platform for AI

**Duration**: 85 hours
**Complexity**: Very High (Data Engineering Heavy)
**Business Impact**: 50% reduction in data engineering time

**Executive Summary**:
Architect a unified data platform supporting both batch and real-time ML workloads, processing 100TB+ daily, with data governance, quality, and feature engineering at scale.

**Business Value**:
- **50% reduction** in data engineering time (self-service for data scientists)
- **99.9% data quality** with automated monitoring
- **60% cost savings** vs separate lake + warehouse (lakehouse architecture)
- **Complete data lineage** and governance
- **10M events/sec** real-time streaming capability

**Learning Outcomes**:
- Design data lakehouse architectures (Delta Lake/Iceberg/Hudi)
- Architect real-time streaming platforms (Kafka, Flink)
- Implement data governance frameworks (catalog, lineage, quality)
- Integrate ML platforms with data platforms
- Build feature engineering infrastructure at scale
- Design data quality frameworks with automation

**Key Technologies**:
- Delta Lake / Apache Iceberg / Apache Hudi (lakehouse)
- Apache Kafka, Apache Flink (streaming)
- Apache Spark, Airflow (batch processing)
- dbt (data transformation)
- Data Catalog (Datahub, Amundsen)
- Feature Store (Feast, Tecton)

**Architecture Artifacts**:
- Data lakehouse architecture (format comparison and selection)
- Real-time streaming architecture handling 10M events/sec
- Data governance framework (catalog, lineage, quality, privacy)
- ML platform integration design
- Feature engineering platform architecture
- Data quality framework with automated monitoring
- 10+ ADRs (lakehouse format, streaming platform, governance approach)

**Prerequisites**:
- Project 301 completion (understand ML platform integration)
- Data engineering fundamentals
- Experience with Spark or similar data processing frameworks

**Best For**:
- Data platform architects
- Those unifying batch and streaming workloads
- Organizations with data quality and governance requirements
- Feature store implementation focus

---

### Project 305: Security and Compliance Framework

**Duration**: 70 hours
**Complexity**: Very High (Compliance Heavy)
**Business Impact**: Compliance certification + risk mitigation

**Executive Summary**:
Create comprehensive security architecture for ML platforms in regulated industries, achieving SOC2, HIPAA, and ISO27001 compliance with zero-trust design.

**Business Value**:
- **Compliance certification** (SOC2, HIPAA, ISO27001) enabling regulated industry customers
- **85% reduction in audit time** through automation
- **Risk mitigation**: Prevent $50M+ in potential fines
- **Zero security incidents** post-deployment
- **Customer trust** and competitive differentiation

**Learning Outcomes**:
- Design zero-trust architectures for ML platforms
- Architect for regulatory compliance (GDPR, HIPAA, SOC2, ISO27001)
- Implement ML-specific security (model security, adversarial defenses)
- Create IAM architecture with fine-grained access control
- Design encryption strategies (at rest, in transit, in use)
- Build incident response frameworks
- Automate compliance and audit procedures

**Key Technologies**:
- Kubernetes security (Pod Security Policies, Network Policies, RBAC)
- HashiCorp Vault (secrets management)
- Cloud KMS (encryption key management)
- SIEM platforms (Splunk, Elastic Security)
- Service mesh (Istio, Linkerd) for zero-trust
- Confidential computing (SGX, SEV)

**Architecture Artifacts**:
- Zero-trust architecture design for ML platform
- Comprehensive compliance framework (200+ controls)
- ML-specific security considerations
- IAM architecture with RBAC and attribute-based access control
- Encryption strategy (at rest, in transit, in use)
- Incident response framework with runbooks
- Security monitoring and SIEM architecture
- Reference Kubernetes security implementation
- 10+ ADRs (zero-trust approach, secrets management, encryption strategy)

**Prerequisites**:
- Project 301 completion (understand ML platform to secure)
- Security fundamentals
- Compliance basics (GDPR, HIPAA, etc.)

**Best For**:
- Healthcare and finance ML platforms
- Organizations requiring compliance certification
- Security architects specializing in ML
- Those implementing zero-trust architectures

---

## Skills Matrix

### Project-by-Project Skills Breakdown

| Skill Category | P301 | P302 | P303 | P304 | P305 | Total Coverage |
|----------------|------|------|------|------|------|----------------|
| **Architecture Design** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Business Cases & ROI** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Expert |
| **Stakeholder Communication** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Expert |
| **Governance & Compliance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Multi-Cloud Architecture** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced |
| **Kubernetes Expertise** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **MLOps & ML Platforms** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Expert |
| **Data Engineering** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Advanced |
| **LLM & GenAI** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Advanced |
| **Security & Zero-Trust** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Cost Optimization** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Expert |
| **HA/DR Planning** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Expert |

**Rating Scale**:
- ⭐⭐⭐⭐⭐ Expert: Mastery-level coverage, can teach others
- ⭐⭐⭐⭐ Advanced: Comprehensive coverage, can implement independently
- ⭐⭐⭐ Intermediate: Good foundation, can contribute with guidance
- ⭐⭐ Basic: Introduced to concepts, requires more practice
- ⭐ Minimal: Tangential mention only

### Core Architect Skills

**Skills Developed Across All Projects**:

1. **Architecture Design**:
   - C4 diagram creation (Context, Container, Component, Deployment)
   - ADR writing and decision documentation
   - Multi-year technology roadmaps
   - Vendor selection frameworks
   - Scalability design (10x, 100x growth)

2. **Business Skills**:
   - Business case development (NPV, IRR, payback period)
   - Cost-benefit analysis for $10M+ investments
   - TCO (Total Cost of Ownership) modeling
   - ROI calculations with sensitivity analysis
   - Executive presentation skills

3. **Governance & Compliance**:
   - Model governance frameworks
   - Data governance and lineage
   - Regulatory compliance (GDPR, HIPAA, SOC2, ISO27001)
   - Responsible AI and ethical AI
   - Audit automation and reporting

4. **Strategic Leadership**:
   - Multi-cloud and hybrid architecture strategies
   - Cost optimization ($5M+ annual savings)
   - Disaster recovery and business continuity
   - FinOps frameworks and cost allocation
   - Build vs buy vs partner decisions

5. **Stakeholder Management**:
   - C-suite communication (CEO, CFO, CTO)
   - Board of directors presentations
   - Cross-functional alignment (Engineering, Product, Legal, Finance)
   - Technical to non-technical translation
   - Consensus building and influence

---

## Learning Path Recommendations

### By Career Goal

**Goal: Transition from Senior Engineer → Architect**

**Recommended Path** (300 hours):
1. **Project 301** (80h) - Foundation for architecture thinking
2. **Project 302** (100h) - Multi-cloud strategy and HA/DR
3. **Project 303** (90h) - Modern LLM architecture
4. **Capstone** (30h) - Apply to your organization

**Focus**: Learn architecture artifacts, business cases, and stakeholder communication.

---

**Goal: ML Platform Architect Specialization**

**Recommended Path** (255 hours):
1. **Project 301** (80h) - Core MLOps platform
2. **Project 303** (90h) - LLM and GenAI platform
3. **Project 304** (85h) - Data platform integration

**Focus**: Deep dive into ML-specific architectures and tooling.

---

**Goal: Enterprise Architect (Multi-Cloud Focus)**

**Recommended Path** (270 hours):
1. **Project 302** (100h) - Multi-cloud architecture
2. **Project 301** (80h) - Platform architecture patterns
3. **Project 305** (70h) - Security and compliance
4. **Capstone** (20h) - Enterprise roadmap

**Focus**: Multi-cloud, HA/DR, cost optimization, enterprise governance.

---

**Goal: Security & Compliance Architect**

**Recommended Path** (235 hours):
1. **Project 305** (70h) - Security and compliance foundation
2. **Project 301** (80h) - Governance frameworks
3. **Project 304** (85h) - Data governance and privacy

**Focus**: Zero-trust, compliance frameworks, security architecture.

---

**Goal: Fast-Track Review (Experienced Architects)**

**Recommended Path** (120 hours):
1. **All project READMEs and business cases** (20h)
2. **Key ADRs from each project** (30h) - 5 per project
3. **Template and framework harvest** (20h)
4. **Deep dive on 2 most relevant projects** (50h)

**Focus**: Extract templates, patterns, and best practices for reuse.

---

### By Available Time

**Full-Time Study** (425 hours over 12 weeks):
- **Weeks 1-2**: Project 301 (80h)
- **Weeks 3-5**: Project 302 (100h)
- **Weeks 6-8**: Project 303 (90h)
- **Weeks 9-11**: Projects 304 & 305 (155h total)
- **Week 12**: Integration and capstone

**Part-Time Study** (15 hours/week over 28 weeks, ~7 months):
- **Weeks 1-6**: Project 301 (80h)
- **Weeks 7-13**: Project 302 (100h)
- **Weeks 14-19**: Project 303 (90h)
- **Weeks 20-25**: Project 304 (85h)
- **Weeks 26-28**: Project 305 (70h)

**Weekend Study** (8 hours/weekend over 53 weekends, ~1 year):
- **Weekends 1-10**: Project 301
- **Weekends 11-23**: Project 302
- **Weekends 24-35**: Project 303
- **Weekends 36-46**: Project 304
- **Weekends 47-53**: Project 305 + integration

---

## Project Dependencies

### Prerequisite Flow

```
┌─────────────────────────────────────────┐
│    Senior Engineer Track Completion      │
│         (or equivalent 5-8 years)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Project 301: Enterprise MLOps       │  ← START HERE (REQUIRED)
│     (Foundation for all other projects)  │
└────────┬────────────────────────────────┘
         │
         ├────────────────────┬────────────────────┬─────────────────┐
         │                    │                    │                 │
         ▼                    ▼                    ▼                 ▼
┌────────────────┐  ┌─────────────────┐  ┌────────────────┐  ┌─────────────────┐
│   Project 302  │  │   Project 303   │  │  Project 304   │  │   Project 305   │
│  Multi-Cloud   │  │  LLM Platform   │  │ Data Platform  │  │    Security     │
│                │  │                 │  │                │  │                 │
│  (Recommended  │  │  (Recommended   │  │  (Recommended  │  │  (Recommended   │
│   2nd or 3rd)  │  │   2nd or 3rd)   │  │   3rd or 4th)  │  │   4th or 5th)   │
└────────────────┘  └─────────────────┘  └────────────────┘  └─────────────────┘
```

### Dependency Details

**Project 301** (Enterprise MLOps Platform):
- **Prerequisites**: None (start here!)
- **Recommended First**: Yes - establishes architecture thinking
- **Provides Foundation For**: All other projects
- **Key Concepts**: ADRs, C4 diagrams, business cases, governance

**Project 302** (Multi-Cloud Infrastructure):
- **Prerequisites**: Project 301 (understand ML platform to distribute across clouds)
- **Builds On**: P301 Kubernetes and platform concepts
- **Recommended Order**: 2nd or 3rd
- **Key Concepts**: Multi-cloud portability, HA/DR, FinOps

**Project 303** (LLM Platform with RAG):
- **Prerequisites**: Project 301 (MLOps foundation)
- **Builds On**: P301 model serving and monitoring
- **Recommended Order**: 2nd or 3rd
- **Key Concepts**: LLM infrastructure, RAG, responsible AI

**Project 304** (Data Platform):
- **Prerequisites**: Project 301 (understand ML platform to feed with data)
- **Builds On**: P301 feature store integration
- **Recommended Order**: 3rd or 4th
- **Key Concepts**: Lakehouse, streaming, data governance

**Project 305** (Security Framework):
- **Prerequisites**: Project 301 (need platform to secure)
- **Builds On**: All previous projects' security considerations
- **Recommended Order**: 4th or 5th (integrates learnings from others)
- **Key Concepts**: Zero-trust, compliance, encryption

---

## Technology Coverage

### Infrastructure & Orchestration

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **Kubernetes** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Terraform** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Expert |
| **Crossplane** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | Advanced |
| **Helm** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Advanced |
| **ArgoCD/GitOps** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | Intermediate |

### ML Platforms & Tools

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **MLflow** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Expert |
| **Kubeflow** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | Advanced |
| **Feast** (Feature Store) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Expert |
| **KServe** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Expert |
| **vLLM** | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | Expert |
| **TensorRT-LLM** | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | Expert |
| **LangChain/LlamaIndex** | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | Advanced |

### Data Platforms

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **Delta Lake** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Expert |
| **Apache Iceberg** | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ | Advanced |
| **Apache Kafka** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Expert |
| **Apache Spark** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Expert |
| **Apache Flink** | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Expert |
| **dbt** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ | Advanced |
| **Datahub/Amundsen** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Expert |

### Cloud Platforms

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **AWS** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Expert |
| **GCP** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Advanced |
| **Azure** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Advanced |

### Security & Compliance

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **HashiCorp Vault** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Istio/Linkerd** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **SIEM** (Splunk/Elastic) | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |
| **Cloud KMS** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expert |

### Monitoring & Observability

| Technology | P301 | P302 | P303 | P304 | P305 | Depth |
|------------|------|------|------|------|------|-------|
| **Prometheus** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Expert |
| **Grafana** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Expert |
| **Jaeger/Tempo** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Advanced |
| **ELK Stack** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Advanced |

**Total Unique Technologies**: 40+

---

## Artifact Types Catalog

### What You'll Create

For each project, you'll study and potentially create:

**Architecture Artifacts** (60% of content):
- **C4 Diagrams** (4 levels per project):
  - Context: System in environment, external users and systems
  - Container: High-level technology choices, applications, databases
  - Component: Internal structure of containers, components and relationships
  - Deployment: Infrastructure topology, networking, deployment

- **Architecture Decision Records (ADRs)** (10+ per project):
  - Technology stack selection
  - Build vs buy decisions
  - Multi-tenancy approach
  - Governance framework design
  - Cost optimization strategies
  - Security and compliance approach

- **Business Cases**:
  - Executive summary (1-page)
  - Problem statement with quantified pain
  - Solution overview
  - Financial analysis (NPV, IRR, payback period, TCO)
  - Risk assessment with mitigation strategies
  - Implementation roadmap

- **Stakeholder Presentations**:
  - Executive presentation (20-30 slides, business-focused)
  - Technical deep-dive (30-40 slides, architecture-focused)
  - Operational overview (runbooks, procedures)

- **Governance Frameworks**:
  - Model governance (approval workflows, audit trails)
  - Data governance (catalog, lineage, quality)
  - Compliance frameworks (GDPR, HIPAA, SOC2, ISO27001)
  - Architecture governance (review processes, standards)

**Reference Implementations** (40% of content):
- **Infrastructure as Code**:
  - Terraform modules (multi-cloud compatible)
  - Cloud-specific configurations (AWS/GCP/Azure)
  - Networking and security configs

- **Kubernetes Manifests**:
  - Deployments, Services, ConfigMaps, Secrets
  - Custom Resource Definitions (CRDs)
  - Helm charts

- **Platform APIs**:
  - REST/gRPC API examples
  - SDKs for data scientists
  - Integration examples

- **Monitoring Configurations**:
  - Prometheus metrics and alerts
  - Grafana dashboards
  - Logging and tracing configs

---

## Career Alignment

### Current Role: AI Infrastructure Architect

**Typical Titles**:
- AI Infrastructure Architect (Level 3)
- ML Platform Architect
- Senior Staff Engineer, ML Infrastructure
- Principal Engineer, MLOps
- Director of ML Infrastructure (smaller companies)

**Responsibilities**:
- Design enterprise ML platforms supporting 100+ teams
- Create multi-year technology roadmaps
- Present architecture to C-suite and board
- Lead technical decision-making across organization
- Mentor senior engineers on architecture thinking
- Drive $5M+ cost optimization initiatives

**Success Metrics**:
- Business value delivered ($10M-$50M NPV)
- Platform adoption (number of teams, models)
- Cost savings (30%+ reduction)
- System reliability (99.95%+ uptime)
- Compliance achievement (SOC2, HIPAA, etc.)
- Team enablement (self-service adoption)

### Next Role: Senior AI Infrastructure Architect

**Typical Titles**:
- Senior AI Infrastructure Architect (Level 4)
- Distinguished Engineer, AI/ML
- VP of ML Infrastructure
- Chief AI Architect
- CTO (AI-focused startups)

**Additional Responsibilities**:
- Enterprise transformation leadership (5-year strategies)
- Board advisory on AI technology
- Industry thought leadership (publications, standards)
- Executive coaching and communication
- Strategic partnership and M&A evaluation
- Multi-organization impact (divisions, acquisitions)

**Preparation via** [Senior Architect Solutions Repository](../ai-infra-senior-architect-solutions/)

### Salary Ranges (US, 2025)

**AI Infrastructure Architect (Level 3)**:
- **Base Salary**: $200K - $300K
- **Total Compensation**: $350K - $600K
  - Base + Bonus (20-30%) + Equity
  - Big Tech (FAANG): $500K-$600K total
  - Unicorn Startups: $400K-$550K total
  - Mid-size companies: $350K-$450K total

**Consulting Rates** (for independent architects):
- **Hourly**: $250 - $500/hour
- **Daily**: $2,000 - $4,000/day
- **Project-based**: $50K - $200K for 3-6 month engagements

**Geographic Multipliers**:
- San Francisco / Seattle: 1.0x (baseline)
- New York: 0.95x
- Austin / Denver: 0.85x
- Remote (US): 0.80-0.90x
- Europe: 0.60-0.75x (€150K-€250K total comp)
- Asia-Pacific: 0.50-0.70x (highly variable)

### Skills Impact on Compensation

**High-Value Skills** (30%+ comp premium):
- Multi-cloud architecture expertise
- LLM/GenAI platform experience
- Proven ROI/cost optimization ($10M+ savings)
- Executive communication and influence
- Published thought leadership

**Emerging High-Demand**:
- GenAI/LLM infrastructure (50%+ demand increase in 2024-2025)
- Responsible AI and governance (compliance mandates)
- FinOps and cloud cost optimization (recession-driven)
- Multi-cloud portability (vendor independence)

---

## Appendix: Quick Reference

### Project Quick Facts

| Project | Hours | Complexity | Business Impact | Best For |
|---------|-------|------------|----------------|----------|
| **301: Enterprise MLOps** | 80h | High | $30M NPV | Platform architects, first project |
| **302: Multi-Cloud** | 100h | Very High | $8M savings | Global orgs, vendor independence |
| **303: LLM Platform** | 90h | Very High | $4.2M savings | GenAI focus, cost optimization |
| **304: Data Platform** | 85h | Very High | 50% time reduction | Data engineering, governance |
| **305: Security** | 70h | Very High | Compliance + risk | Regulated industries, zero-trust |
| **TOTAL** | **425h** | - | **$50M+ value** | Complete mastery |

### Artifact Deliverable Count

| Artifact Type | Per Project | Total (5 projects) |
|---------------|-------------|-------------------|
| **C4 Diagram Sets** | 1 (4 levels) | 5 sets (20 diagrams) |
| **ADRs** | 10+ | 50+ |
| **Business Cases** | 1 complete | 5 |
| **Executive Presentations** | 1 (20-30 slides) | 5 |
| **Tech Deep-Dives** | 1 (30-40 slides) | 5 |
| **Governance Frameworks** | 2-3 | 10-15 |
| **Reference Implementations** | 1 complete | 5 |

### Study Time Estimates

| Activity | Time per Project | Total (5 projects) |
|----------|------------------|-------------------|
| **Reading/Understanding** | 15-20h | 75-100h |
| **ADR Analysis** | 10-15h | 50-75h |
| **Diagram Study** | 5-8h | 25-40h |
| **Business Case Analysis** | 8-10h | 40-50h |
| **Reference Implementation** | 10-15h | 50-75h |
| **Application to Your Org** | 20-30h | 100-150h |
| **TOTAL** | **~85h** | **425h** |

---

**Questions or feedback?** Open an issue or email: ai-infra-curriculum@joshua-ferguson.com

**Ready to start?** → [QUICK_START_GUIDE.md](./QUICK_START_GUIDE.md) or [Project 301](./projects/project-301-enterprise-mlops/README.md)

---

**Last Updated**: October 25, 2025
**Version**: 1.0.0
**Maintained By**: AI Infrastructure Curriculum Team
