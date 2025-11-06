# Module Roadmap

> Based on the legacy AI Infrastructure Engineer data pipeline module with updated alignment and validation.

## Module Overview

- **Module ID**: MOD-105
- **Module Title**: Data Pipelines & Orchestration
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 45 hours (15 lecture, 20 lab, 8 project, 2 assessment)
- **Prerequisites**: MOD-101, MOD-102, MOD-004, MOD-008
- **Next Module(s)**: MOD-106, PROJ-202

## Cross-Role Progression

- **Builds On** (modules/roles): Junior ML basics (MOD-004) and Databases & SQL (MOD-008)
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Airflow quick-start DAG reused as onboarding exercise
- **Differentiators** (role-specific emphasis): Production DAG design, data lineage, large-scale processing

## Learning Objectives

- Architect ML-friendly data pipelines with Airflow and complementary orchestration tooling.
- Implement data versioning, lineage tracking, and governance suitable for regulated environments.
- Integrate distributed processing (Spark) and streaming (Kafka) into ML workflows.
- Build automated data quality and validation checks with alerting.
- Package pipelines for deployment with infrastructure-as-code and CI/CD.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Data pipeline engineering | Proficient | Airflow DAG + monitoring review | AI Infrastructure Engineer |
| Data governance | Working | Lineage and validation documentation | AI Infrastructure Engineer |
| Automation & tooling | Working | CI/CD pipeline for DAG deployment | AI Infrastructure Engineer |

## Content Outline

1. **Pipeline Architecture Patterns** – batch vs streaming, ML lifecycle integration.
2. **Airflow Fundamentals** – DAG structure, operators, scheduling, dependency management.
3. **Data Versioning & Lineage** – DVC, LakeFS, metadata tracking, lineage visualization.
4. **Distributed Processing** – Spark jobs, resource sizing, optimization for ML feature pipelines.
5. **Streaming & Real-Time** – Kafka basics, consumer groups, ML inference triggers.
6. **Quality & Governance** – Great Expectations checks, anomaly detection, incident response.
7. **Operationalization** – CI/CD for DAGs, infrastructure packaging, observability hooks.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Author production-grade Airflow DAG | DAG runs end-to-end with retries & SLAs | Airflow unit test suite |
| Lab 2 | Implement data versioning & lineage | Versioned dataset stored; lineage graph exported | Data governance checklist |
| Lab 3 | Streaming feature pipeline prototype | Kafka topic processing with monitoring dashboards | Integration tests |
| Assessment | Pipeline architecture quiz | ≥75% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Airflow DAGs, Spark job templates, Great Expectations suite, quiz key
- **Repository Strategy**: `per_role` (see `curriculum/repository-strategy.yaml`)
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-105-data-pipelines/solutions`
- **Validation Status**: Pending execution of DAG unit tests and data quality checks in CI

## Resource Plan

- **Primary References**:
  - Module README in `modules/ai-infrastructure-engineer/module-105-data-pipelines`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-105-data-pipelines`
- **Supplemental Resources**:
  - Apache Airflow production guide, LakeFS documentation
  - Great Expectations best practices
- **Tooling Requirements**:
  - Airflow 2.x, Spark cluster or local mode, Kafka (local stack), Great Expectations CLI

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Foundational ML knowledge (MOD-004), databases (MOD-008), cloud provisioning (MOD-102)
- **Downstream Outputs**: Feeds MLOps module (MOD-106) and Project PROJ-202
- **Risks / Mitigations**: Complexity creep—enforce incremental milestone reviews before adding streaming components.
