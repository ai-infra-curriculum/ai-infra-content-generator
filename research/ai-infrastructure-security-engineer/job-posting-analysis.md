# Job Posting Analysis — AI Infrastructure Security Engineer

> Sample set: 24 postings (2024–2025) from hyperscalers, foundation model companies, fintech, healthcare, autonomous systems, and SaaS platforms expanding AI infrastructure.

## Summary

- **Regions**: SF Bay, Seattle, Austin, NYC, Toronto, London, Dublin, Berlin, Bangalore, Singapore.  
- **Industries**: Cloud platform, AI/ML product, financial services, healthcare, autonomous vehicles, industrial IoT.  
- **Org Size**: 1k – 150k+ employees; security engineer headcount typically embedded within platform or security orgs supporting 5–10 squads.

## Aggregate Skill Frequency

| Skill | Frequency | % of Postings | Category | Notes |
|-------|-----------|---------------|----------|-------|
| Zero-trust / identity architecture | 22 | 92% | Core | Strong emphasis on workload identity, segmentation, continuous verification |
| Secrets management & key management | 20 | 83% | Core | Vault/KMS, service identities, rotation automation |
| Runtime security & detection | 20 | 83% | Core | eBPF, container security, SIEM/SOAR integrations |
| DevSecOps / policy-as-code | 19 | 79% | Core | CI/CD hardening, admission controllers, guardrails |
| ML-specific threat modeling & adversarial defense | 17 | 71% | Core | Poisoning, evasion, model theft defenses |
| Compliance automation & evidence pipelines | 16 | 67% | Important | SOC2, ISO, HIPAA, EU AI Act readiness |
| Supply chain security (SBOM, attestation) | 15 | 63% | Important | Sigstore, SLSA, artifact provenance |
| Incident response / SOC integration | 14 | 58% | Important | On-call with SOC, playbooks, executive comms |

## Technology & Tool Mentions

| Category | Tool / Platform | Frequency | Notes |
|----------|-----------------|-----------|-------|
| Identity & Secrets | AWS/GCP/Azure IAM, SPIFFE/SPIRE, HashiCorp Vault, AAD Workload ID | 19 | Identity-first zero-trust focus |
| Network & Policy | Istio, Linkerd, Calico, OPA, Kyverno, Cloud Custodian | 18 | Service mesh & policy-as-code |
| Runtime Security | Falco, eBPF, Aqua, Wiz, GuardDuty, SentinelOne, Datadog | 17 | Integration with SIEM/SOAR | 
| Supply Chain | Sigstore/Cosign, in-toto, SLSA frameworks, Syft/Grype | 15 | Artifact attestation, SBOM automation |
| Compliance Automation | Drata, Vanta, JupiterOne, custom control pipelines | 14 | Evidence orchestration |
| Adversarial/ML Security | ART, CleverHans, SecML, custom red-team harnesses | 13 | Testing + defenses |

## Responsibility Themes

| Theme | Representative Responsibilities | Frequency |
|-------|--------------------------------|-----------|
| Zero-trust & segmentation | Identity strategy, network segmentation, access controls | 22 |
| ML pipeline & runtime hardening | Secure data/model pipelines, runtime detection, secrets | 21 |
| DevSecOps automation | Policy-as-code, CI/CD guardrails, supply chain security | 19 |
| Compliance & governance | Control mapping, evidence automation, regulator readiness | 17 |
| Adversarial defense | Testing, mitigation, monitoring for ML threats | 17 |
| Incident response & SOC integration | Runbooks, detection engineering, executive comms | 15 |

## Hiring Signals & Requirements

- Deep **cloud security and DevSecOps** expertise expected; many postings require prior ownership of large-scale security automation efforts.  
- Strong emphasis on **collaboration with platform/MLOps** teams—security engineers must code, design architecture, and lead multi-team initiatives.  
- ML-specific knowledge increasingly highlighted: adversarial ML, data privacy, model protection, safety/regulatory experience.  
- **Hands-on proficiency** (Terraform, Python, Rego, container security) demanded even for senior-level roles.  
- Experience preparing for **AI governance and compliance** programs (EU AI Act, NIST AI RMF, ISO/IEC 42001) emerging as a differentiator.

## Source References

- `/home/claude/ai-infrastructure-project/research/role-analysis.json` (Security Engineer sections)  
- `/home/claude/ai-infrastructure-project/research/SKILLS_MATRIX_12_ROLES.md`  
- Job boards: AWS, Google, Microsoft, Anthropic, OpenAI, Scale AI, JPMorgan, Capital One, Mayo Clinic, Tesla, Snowflake.
