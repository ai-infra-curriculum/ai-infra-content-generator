# Module 04: Kubernetes Fundamentals

**Duration:** 60 hours
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Modules 01, 02, and 03 complete

## Module Overview

Kubernetes (K8s) is the industry-standard platform for container orchestration. This module provides comprehensive coverage of Kubernetes fundamentals with a focus on deploying and managing ML workloads.

You'll learn to deploy ML models on Kubernetes, manage compute resources (including GPUs), implement auto-scaling, and troubleshoot common issues. By the end, you'll be proficient in operating production Kubernetes clusters for ML applications.

## Why Kubernetes for ML?

### The Challenge: Container Orchestration at Scale

**Without Kubernetes:**
- Manually start/stop containers across 100 machines
- No automatic recovery if a container crashes
- Manual load balancing across instances
- Complex networking between services
- Manual scaling up/down
- No built-in service discovery

**With Kubernetes:**
- Declare desired state, K8s maintains it automatically
- Self-healing: restarts failed containers
- Built-in load balancing and service discovery
- Automatic scaling based on metrics
- Rolling updates and rollbacks
- Standardized deployment across environments

### Real-World Impact

**Example: Netflix**
- 1,000+ microservices on Kubernetes
- Automatic scaling from 100 to 10,000 pods during peak hours
- Self-healing recovers from node failures in seconds

**Example: Spotify**
- 200+ Kubernetes clusters globally
- GPU scheduling for ML model training
- ~80% cost reduction through efficient resource utilization

## Learning Outcomes

By the end of this module, you will be able to:

1. **Understand K8s Architecture** - Explain control plane, nodes, pods, and core concepts
2. **Deploy Applications** - Deploy ML models as Kubernetes Deployments
3. **Manage Resources** - Use ConfigMaps, Secrets, Persistent Volumes
4. **Configure Networking** - Set up Services, Ingress, load balancing
5. **Schedule GPU Workloads** - Deploy GPU-accelerated ML inference and training
6. **Use Helm Charts** - Package and deploy applications with Helm
7. **Implement Auto-Scaling** - Configure HPA (Horizontal Pod Autoscaler)
8. **Monitor Clusters** - Set up basic monitoring with kubectl and metrics
9. **Troubleshoot Issues** - Debug pods, services, networking problems
10. **Apply Best Practices** - Security, resource limits, health checks

## Module Structure

### Lesson 01: Kubernetes Introduction (6 hours)
**File:** `01-k8s-introduction.md`

- What is Kubernetes and why it matters
- Kubernetes vs Docker Compose vs Docker Swarm
- K8s history and ecosystem (CNCF)
- When to use Kubernetes (and when not to)
- Setting up local cluster (minikube, kind, k3s)
- **Hands-on:** Deploy first application to K8s

### Lesson 02: Kubernetes Architecture (8 hours)
**File:** `02-k8s-architecture.md`

- Control Plane components (API server, etcd, scheduler, controller manager)
- Node components (kubelet, kube-proxy, container runtime)
- Add-ons (DNS, Dashboard, monitoring)
- How K8s makes scheduling decisions
- Understanding the API server and kubectl
- **Hands-on:** Explore cluster components

### Lesson 03: Core Resources - Pods and Namespaces (7 hours)
**File:** `03-core-resources.md`

- Pods: the smallest deployable unit
- Multi-container pods and sidecar pattern
- Init containers
- Namespaces for logical isolation
- Labels and selectors
- Resource requests and limits
- **Hands-on:** Deploy ML inference pod

### Lesson 04: Deployments and Services (8 hours)
**File:** `04-deployments-services.md`

- Deployments for managing replicas
- ReplicaSets (managed by Deployments)
- Rolling updates and rollbacks
- Services: ClusterIP, NodePort, LoadBalancer
- Service discovery and DNS
- Endpoints and EndpointSlices
- **Hands-on:** Deploy scalable ML API with load balancing

### Lesson 05: Networking and Ingress (7 hours)
**File:** `05-networking-ingress.md`

- Kubernetes networking model
- CNI plugins (Calico, Cilium, Flannel)
- Network Policies for security
- Ingress controllers (Nginx, Traefik)
- TLS termination and SSL certificates
- Exposing services to the internet
- **Hands-on:** Set up Ingress for multiple services

### Lesson 06: Storage and Persistence (7 hours)
**File:** `06-storage-persistence.md`

- Volumes and PersistentVolumes (PV)
- PersistentVolumeClaims (PVC)
- StorageClasses for dynamic provisioning
- StatefulSets for stateful applications
- Volume types (hostPath, NFS, cloud storage)
- Managing model artifacts and datasets
- **Hands-on:** Deploy model with persistent storage

### Lesson 07: Configuration and Secrets (6 hours)
**File:** `07-configuration-secrets.md`

- ConfigMaps for configuration data
- Secrets for sensitive information
- Environment variables
- Volume mounts for config/secrets
- External secret management (HashiCorp Vault)
- Best practices for secret management
- **Hands-on:** Configure ML app with ConfigMaps and Secrets

### Lesson 08: Helm Package Manager (6 hours)
**File:** `08-helm-package-manager.md`

- What is Helm and why use it?
- Helm charts structure
- Installing charts from repositories
- Creating custom charts
- Templating and values files
- Helm for ML deployments
- **Hands-on:** Package ML application as Helm chart

### Lesson 09: GPU Scheduling (5 hours)
**File:** `09-gpu-scheduling.md`

- GPU device plugin for Kubernetes
- Requesting GPU resources
- GPU sharing strategies
- Multi-GPU pods
- GPU node pools
- Troubleshooting GPU issues
- **Hands-on:** Deploy GPU-accelerated model

### Lesson 10: Monitoring and Observability (6 hours)
**File:** `10-monitoring-observability.md`

- kubectl commands for monitoring
- Metrics Server
- Resource usage monitoring
- Logs aggregation
- Health checks (liveness, readiness, startup probes)
- Prometheus basics for K8s
- **Hands-on:** Set up monitoring for ML workload

## Hands-On Activities

### Activity 1: Deploy First ML Model to Kubernetes
**Duration:** 2 hours
**Deliverable:** Deployment running 3 replicas of ML inference service

### Activity 2: Set Up Complete ML Stack
**Duration:** 4 hours
**Deliverable:** Model server + Redis + PostgreSQL with networking

### Activity 3: Implement Auto-Scaling
**Duration:** 2 hours
**Deliverable:** HPA scaling pods based on CPU/memory

### Activity 4: Configure Ingress with TLS
**Duration:** 3 hours
**Deliverable:** Public HTTPS endpoint for ML API

### Activity 5: Deploy GPU Workload
**Duration:** 3 hours
**Deliverable:** GPU-accelerated inference with resource limits

## Advanced Exercises

After mastering the foundational Kubernetes concepts through lessons and hands-on activities, challenge yourself with these production-grade exercises:

### Exercise 04: Kubernetes Cluster Autoscaler with Custom Metrics (35-43 hours)
Build a production-grade cluster autoscaling system that scales nodes based on custom metrics.

**Location:** `exercises/exercise-04-k8s-cluster-autoscaler/`
**Prerequisites:** Completed all lessons, understanding of Prometheus, Custom Metrics API
**Skills:** Advanced autoscaling (HPA, VPA, Cluster Autoscaler), custom metrics, cost optimization

**Topics:**
- Horizontal Pod Autoscaler (HPA) with custom metrics
- Vertical Pod Autoscaler (VPA)
- Cluster Autoscaler configuration
- GPU utilization metrics
- Model inference queue depth monitoring
- Cost-optimized scaling strategies
- Multi-tenancy and fair resource allocation

### Exercise 05: Service Mesh Observability (estimated 35-45 hours)
Implement a complete service mesh for ML microservices with advanced observability.

**Location:** `exercises/exercise-05-service-mesh-observability/`
**Prerequisites:** Completed Exercise 04, understanding of service mesh concepts
**Skills:** Istio/Linkerd, distributed tracing, service mesh patterns

**Topics:**
- Service mesh installation and configuration
- Traffic management (canary, blue-green)
- mTLS and security policies
- Distributed tracing with Jaeger/Zipkin
- Advanced observability and metrics
- Failure injection and chaos engineering

### Exercise 06: Kubernetes Operator Framework (estimated 35-45 hours)
Build a custom Kubernetes operator for managing ML model lifecycle.

**Location:** `exercises/exercise-06-k8s-operator-framework/`
**Prerequisites:** Completed Exercise 05, Go or Python programming
**Skills:** Operator pattern, Custom Resource Definitions (CRDs), controller development

**Topics:**
- Custom Resource Definitions (CRDs)
- Operator SDK and controller-runtime
- Reconciliation loops
- ML model lifecycle management
- Automated training job scheduling
- Model versioning and rollback
- Integration with ML registries

## Assessments

### Quiz: Kubernetes Architecture and Concepts
**Location:** `quiz.md`
**Questions:** 25 multiple choice and short answer
**Passing Score:** 70% (18/25 correct)
**Topics:** Architecture, resources, networking, storage, GPU scheduling

### Practical: Deploy Production-Ready ML System on Kubernetes
**Objective:** Deploy complete ML system with best practices

**Requirements:**
1. Deployment with 3+ replicas
2. Service with load balancing
3. Ingress with SSL/TLS
4. ConfigMaps for configuration
5. Secrets for sensitive data
6. PersistentVolume for model storage
7. Resource limits defined
8. Health checks configured
9. HPA for auto-scaling
10. Monitoring and logging

**Submission:** Kubernetes manifests, README, and demo

## Prerequisites Check

Before starting Module 04:

- [ ] Completed Modules 01-03
- [ ] Understand Docker containers well
- [ ] Familiarity with YAML syntax
- [ ] kubectl installed locally
- [ ] Local Kubernetes cluster (minikube, kind, or Docker Desktop K8s)
- [ ] Basic understanding of networking concepts
- [ ] (Optional) Access to cloud Kubernetes (EKS, GKE, AKS)

### Installation Verification

```bash
# Verify kubectl installation
kubectl version --client

# Verify local cluster (if using minikube)
minikube status

# Or verify Docker Desktop K8s
kubectl cluster-info

# Verify Helm (optional, will install in lesson)
helm version
```

## Key Concepts Covered

### Kubernetes Architecture
- Control plane and worker nodes
- API server as central component
- etcd for cluster state
- Scheduler and controller manager roles

### Core Workload Resources
- Pods, ReplicaSets, Deployments
- StatefulSets for stateful apps
- DaemonSets for node-level services
- Jobs and CronJobs

### Networking
- Services (ClusterIP, NodePort, LoadBalancer)
- Ingress for HTTP routing
- Network Policies for security
- DNS and service discovery

### Storage
- Volumes, PersistentVolumes, PersistentVolumeClaims
- StorageClasses and dynamic provisioning
- StatefulSets for data persistence

### Configuration
- ConfigMaps and Secrets
- Environment variables
- Volume mounts

### Advanced Topics
- GPU scheduling and device plugins
- Helm for package management
- Auto-scaling (HPA, VPA)
- Monitoring and observability

## Tools and Technologies

### Required
- **kubectl** - Kubernetes command-line tool
- **Local Kubernetes cluster** - minikube, kind, or Docker Desktop
- **Docker** - For building container images

### Recommended
- **Helm** - Kubernetes package manager
- **k9s** - Terminal-based Kubernetes UI
- **kubectx/kubens** - Context and namespace switching
- **stern** - Multi-pod log tailing

### Optional
- **Lens** - Kubernetes IDE (GUI)
- **Octant** - Web-based K8s dashboard
- **Prometheus** - Monitoring (will cover basics)

## Common Pitfalls and Solutions

### Pitfall 1: Pod Not Starting
**Symptoms:** CrashLoopBackOff, ImagePullBackOff, Pending status
**Solution:** Check logs with `kubectl logs`, describe pod with `kubectl describe pod`

### Pitfall 2: Service Not Accessible
**Symptoms:** Cannot connect to service
**Solution:** Verify service selector matches pod labels, check endpoints

### Pitfall 3: Insufficient Resources
**Symptoms:** Pods stuck in Pending state
**Solution:** Check node resources with `kubectl top nodes`, adjust resource requests

### Pitfall 4: Configuration Issues
**Symptoms:** Application errors, missing environment variables
**Solution:** Verify ConfigMaps and Secrets are created and mounted correctly

### Pitfall 5: GPU Not Detected
**Symptoms:** CUDA errors in pods
**Solution:** Ensure GPU device plugin installed, verify node labels, check resource requests

## Real-World Applications

### Use Case 1: Scalable Model Serving
**Challenge:** Serve ML models handling variable traffic (1K to 100K requests/min)
**Solution:** K8s Deployment with HPA, automatically scales pods based on demand
**Benefit:** Cost-effective, handles spikes automatically

### Use Case 2: Multi-Model Platform
**Challenge:** Serve 50 different models with different resource needs
**Solution:** Each model as separate Deployment, K8s schedules efficiently
**Benefit:** Resource isolation, independent scaling

### Use Case 3: Training Job Orchestration
**Challenge:** Run 100s of training experiments efficiently
**Solution:** Kubernetes Jobs for training, GPU scheduling
**Benefit:** Efficient GPU utilization, automatic retries on failure

### Use Case 4: CI/CD for ML Models
**Challenge:** Automate model testing and deployment
**Solution:** GitOps with K8s, rolling updates, automatic rollback
**Benefit:** Safe deployments, easy rollback, audit trail

## Success Criteria

You've mastered Module 04 when you can:

- [ ] Explain Kubernetes architecture and core components
- [ ] Deploy applications using Deployments and Services
- [ ] Configure networking with Services and Ingress
- [ ] Manage configuration with ConfigMaps and Secrets
- [ ] Set up persistent storage with PVs and PVCs
- [ ] Use Helm to package and deploy applications
- [ ] Schedule GPU workloads efficiently
- [ ] Implement auto-scaling with HPA
- [ ] Monitor and troubleshoot K8s applications
- [ ] Apply security and resource best practices

## Time Estimates

| Component | Estimated Hours |
|-----------|----------------|
| Lessons (10) | 60 hours |
| Exercises (5) | 14-18 hours |
| Quiz | 1-2 hours |
| Practical Assessment | 6-8 hours |
| **Total** | **81-88 hours** |

**Recommended Pace:**
- Part-time (10h/week): 8-9 weeks
- Full-time (40h/week): 2 weeks

## Resources

**Official Documentation:**
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes API Reference](https://kubernetes.io/docs/reference/)
- [Helm Documentation](https://helm.sh/docs/)

**Additional Resources:**
See `resources.md` for comprehensive list of:
- Books and courses
- Video tutorials
- Interactive labs
- Community forums
- Best practices guides

## What's Next?

After completing Module 04:

1. **Complete the quiz** with ≥70% score
2. **Finish the practical assessment** - Deploy production ML system
3. **Choose your path:**
   - **Module 05: Data Pipelines and Orchestration** - Airflow, data workflows
   - **Module 06: MLOps and Experiment Tracking** - MLflow, model registry
   - **Advanced: Kubernetes Operators** - Build custom controllers

## Tips for Success

1. **Practice with local cluster first** - Get comfortable before cloud clusters
2. **Master kubectl** - Learn the command-line tool deeply
3. **Use `kubectl explain`** - Built-in documentation for resources
4. **Read official docs** - Kubernetes docs are excellent
5. **Start simple, add complexity** - Don't try to learn everything at once
6. **Use labels effectively** - Critical for organizing resources
7. **Troubleshoot systematically** - Logs → describe → events → debug
8. **Learn YAML well** - You'll write a lot of it
9. **Experiment and break things** - Local clusters are for learning
10. **Join K8s communities** - Learn from experienced practitioners

---

**Ready to orchestrate at scale?** Start with [Lesson 01: Kubernetes Introduction](./01-k8s-introduction.md)

**Need Kubernetes installed?** Check [Installation Guide](https://kubernetes.io/docs/tasks/tools/)

**Have questions?** Open an issue or discussion in the GitHub repository!
