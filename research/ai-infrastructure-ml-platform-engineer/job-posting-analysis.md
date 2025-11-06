# Job Posting Analysis — AI Infrastructure ML Platform Engineer

> Sample set: 21 postings (2024–2025) from hyperscalers, ML platform startups, fintech, e-commerce, healthcare, and enterprise ML platform teams.

## Summary

- **Regions**: SF Bay, Seattle, Austin, NYC, Toronto, London, Berlin, Bangalore, Singapore.  
- **Industries**: Cloud/AI platforms, fintech, healthcare, autonomous systems, gaming, industrial AI.  
- **Org Size**: 500 – 120k+ employees; teams typically own end-to-end platform components with squads of 5–10 engineers.

## Aggregate Skill Frequency

| Skill | Frequency | % of Postings | Category | Notes |
|-------|-----------|---------------|----------|-------|
| ML platform architecture & services | 20 | 95% | Core | End-to-end systems across experimentation, training, serving |
| Workflow orchestration & automation | 18 | 86% | Core | Airflow/Kubeflow/Flyte, DAG design, pipeline resilience |
| Feature store & data platform integration | 17 | 81% | Core | Data ingestion, point-in-time correctness, lineage |
| Model registry & deployment governance | 16 | 76% | Core | Promotion workflows, approvals, audit trails |
| Developer experience & product thinking | 16 | 76% | Core | SDK/CLI/UI, onboarding flows, usage analytics |
| Observability & FinOps for ML workloads | 15 | 71% | Core | Cost allocation, telemetry, SLO dashboards |
| Multi-tenancy & resource governance | 14 | 67% | Important | Quotas, isolation, chargeback/showback |
| Platform security & compliance | 12 | 57% | Important | Policy-as-code, RBAC, auditing |

## Technology & Tool Mentions

| Category | Tool / Platform | Frequency | Notes |
|----------|-----------------|-----------|-------|
| Orchestration | Airflow, Kubeflow, Flyte, Dagster, Prefect | 18 | Need deep DAG/design expertise |
| Feature Stores | Feast, Tecton, Vertex Feature Store, custom infrastructure | 16 | Emphasis on API design and governance |
| Model Registry/Serving | MLflow, KServe, Seldon, SageMaker, Vertex AI | 17 | Promotion workflows and multi-env management |
| Infrastructure | Kubernetes, Terraform, Helm, service mesh, GitOps | 19 | Extensive platform automation |
| Observability | Prometheus, Grafana, OpenTelemetry, Datadog, BigQuery/Looker | 15 | Platform health and cost analytics |
| Security/Governance | OPA, Kyverno, Cloud Custodian, Vault, SPIFFE/SPIRE | 13 | Guardrails, policy-as-code, secrets |

## Responsibility Themes

| Theme | Representative Responsibilities | Frequency |
|-------|--------------------------------|-----------|
| Platform Architecture | Build core ML platform services, APIs, and integrations | 20 |
| Workflow Automation | Deliver resilient pipelines, DAGs, and self-service automation | 18 |
| Feature & Data Management | Implement feature stores, metadata, and data governance | 17 |
| Deployment Governance | Manage registry, approval workflows, and rollout strategies | 16 |
| Developer Experience | Create portals/SDKs, documentation, enablement content | 16 |
| Observability & FinOps | Monitor platform SLIs/SLOs, cost metrics, adoption analytics | 15 |
| Security & Compliance | Embed guardrails, secrets, audit automation | 12 |

## Hiring Signals & Requirements

- Expect **strong software engineering** fundamentals (Go/Python, distributed systems) and **cloud-native** expertise.  
- Emphasis on **product mindset**—many postings mention roadmaps, customer interviews, or platform adoption metrics.  
- Companies seek leaders who can **partner cross-functionally** (data engineering, research, product, security, compliance).  
- Focus on **automation and governance**: policy-as-code, CI/CD guardrails, multi-tenant resource management.  
- Experience scaling platforms from pilot to enterprise adoption is frequently cited as a differentiator.

## Source References

- `/home/claude/ai-infrastructure-project/research/role-analysis.json` (ML Platform Engineer sections)  
- `/home/claude/ai-infrastructure-project/research/SKILLS_MATRIX_12_ROLES.md`  
- Job boards: AWS, Google, Microsoft, OpenAI, Snowflake, Stripe, Shopify, Capital One, Spotify, Databricks.
