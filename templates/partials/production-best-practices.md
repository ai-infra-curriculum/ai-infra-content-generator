# Partial Template: Production Best Practices Section

> Include in lecture notes or project documentation to cover production-readiness topics.  
> Usage (in markdown): `!INCLUDE templates/partials/production-best-practices.md`

## Production Best Practices

### Security
- Enforce least-privilege access for services and pipelines
- Rotate secrets via managed secret stores (e.g., AWS Secrets Manager, HashiCorp Vault)
- Conduct dependency vulnerability scans on every build

### Reliability
- Implement health probes and circuit breakers for critical services
- Use infrastructure-as-code to version deployments and rollbacks
- Add runbooks and alerts for the top 5 failure scenarios

### Observability
- Standardize logging format and centralize dashboards
- Track golden signals (latency, traffic, errors, saturation)
- Enable tracing for cross-service requests

### Cost Management
- Right-size compute/storage and set budgets with alerts
- Schedule non-production environments to shut down automatically
- Measure feature cost impact as part of releases
