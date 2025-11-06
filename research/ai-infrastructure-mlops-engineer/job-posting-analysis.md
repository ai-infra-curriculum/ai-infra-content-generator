# Job Posting Analysis — AI Infrastructure MLOps Engineer

## Sample Employers & Focus Areas
- **Platform-first companies** (Databricks, Snowflake, Amazon) emphasize hybrid cloud MLOps platforms, GitOps pipelines, and MLflow/Kubeflow expertise.
- **Enterprise data science teams** (Capital One, Walmart, JP Morgan) require regulated model governance, lineage, and automated compliance evidence.
- **Product-led organizations** (Spotify, Airbnb, Doordash) highlight rapid experiment velocity, feature store integration, and developer experience tooling.

## Recurring Requirements
- Own CI/CD pipelines for training and inference workloads (GitHub Actions, GitLab CI, Argo).
- Implement model monitoring, drift detection, and automated retraining workflows (Evidently, Arize, custom observability).
- Standardize model registries, artifact lineage, and promotion workflows (MLflow, SageMaker, Vertex AI).
- Enforce data validation, quality gates, and approval checkpoints before production deployment.
- Align with security/governance teams on policy-as-code, audit trails, and responsible AI guardrails.

## Tooling & Infrastructure Expectations
- Kubernetes + service mesh for model serving at scale.
- Terraform/Pulumi for reproducible infrastructure; Helm/ArgoCD for GitOps.
- Feature stores (Feast, Tecton) and experimentation platforms (Weights & Biases, Neptune).
- Observability stack: Prometheus + Grafana, Datadog, or OpenTelemetry for model/service metrics.
- Message queues/streaming (Kafka, Pub/Sub) to support online inference and feedback loops.

## Skill Gaps Highlighted by Employers
- Combining ML experimentation tooling with traditional DevOps reliability guardrails.
- Holistic governance (model risk, responsible AI, audit evidence) without slowing delivery.
- Productionizing LLM workloads with latency, cost, and safety constraints.
- Cross-functional influence—bridging data science, SRE, security, and product stakeholders.

## Curriculum Implications
- Map foundational content to earlier tracks (engineer/senior) to avoid duplication.
- Emphasize automation, observability, and governance patterns that integrate with ML Platform Engineer assets.
- Capture advanced LLMOps and compliance practices to align with architect/principal tracks.
