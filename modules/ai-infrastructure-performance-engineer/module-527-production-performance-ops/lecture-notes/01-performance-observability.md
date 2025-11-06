# Lecture 01 · Performance Observability

## Objectives
- Design observability dashboards encompassing GPU metrics, latency, throughput, and cost indicators.
- Integrate data sources (DCGM, Prometheus, application metrics) for unified performance visibility.
- Define SLOs/SLA agreements and error budgets for inference workloads.

## Key Topics
1. **Metric Inventory** – GPU utilization, memory bandwidth, latency percentiles, error rates, cost per request.
2. **Tooling Integration** – DCGM exporters, Prometheus scraping, Grafana dashboards, Looker/Tableau overlays.
3. **SLO Design** – defining targets, error budgets, alert thresholds, burn-rate policies.
4. **Collaboration** – aligning with MLOps, platform, and leadership reporting needs.

## Activities
- Create a metrics-to-dashboard mapping for a sample inference service.
- Configure SLO thresholds and alert routes based on real or simulated workloads.
- Produce dashboard mock-ups referencing FinOps requirements.
