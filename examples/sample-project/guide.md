# Sample Project: Observability Rollout

## Objective

Design and implement an observability rollout for a Python microservice stack, delivering dashboards, alerts, and runbooks.

## Deliverables

1. **Architecture Diagram** – Telemetry pipeline showing collectors, backends, and visualization tools.
2. **Implementation Guide** – Step-by-step instructions for instrumenting services and deploying collectors.
3. **Dashboard Pack** – Grafana dashboards (JSON) for latency, error rate, and saturation.
4. **SLO Document** – Service Level Objectives with alert rules.
5. **Runbook** – Incident response steps for latency regression.

## Project Phases

1. **Discovery**
   - Audit current instrumentation.
   - Identify gaps against observability best practices.
2. **Design**
   - Choose exporters (OTLP HTTP) and backend (Tempo/Loki/Prometheus).
   - Define data retention and sampling strategy.
3. **Implementation**
   - Instrument FastAPI services with OpenTelemetry.
   - Deploy collectors via Helm chart.
   - Configure alerts in Grafana Alerting.
4. **Validation**
   - Run load tests to generate telemetry.
   - Verify dashboards and alert firing.

## Assessment Criteria

| Category | Weight | Expectations |
|----------|--------|--------------|
| Architecture | 25% | Accurate pipeline diagram with scalability considerations. |
| Implementation | 35% | Working instrumentation, collectors, and dashboards. |
| Reliability | 20% | SLOs and alerts aligned with business impact. |
| Documentation | 20% | Clear README, runbook, and deployment instructions. |

## Stretch Goals

- Multi-tenant telemetry (prod vs staging).
- Cost optimization plan for storage and retention.
- Automated canary analysis using traces + metrics.
