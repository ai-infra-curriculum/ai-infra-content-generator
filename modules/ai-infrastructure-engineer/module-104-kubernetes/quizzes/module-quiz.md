# Module 04: Kubernetes for ML Infrastructure - Quiz

**Time Limit:** 35 minutes
**Passing Score:** 80% (24/30 questions)
**Coverage:** All Kubernetes lessons

---

## Section 1: Kubernetes Fundamentals (6 questions)

### Q1. What is Kubernetes?

a) A container runtime
b) An container orchestration platform
c) A cloud provider
d) A programming language

**Answer:** B

---

### Q2. What is a Pod in Kubernetes?

a) A cluster of nodes
b) Smallest deployable unit, can contain one or more containers
c) A storage volume
d) A network policy

**Answer:** B

---

### Q3. What is the purpose of a Kubernetes Service?

a) Train ML models
b) Provide stable networking endpoint for Pods
c) Store data
d) Monitor clusters

**Answer:** B

---

### Q4. Which Kubernetes object manages a set of identical Pods?

a) Service
b) Deployment
c) ConfigMap
d) Secret

**Answer:** B

---

### Q5. What is kubectl?

a) Kubernetes cluster
b) Command-line tool for interacting with Kubernetes
c) A container runtime
d) A monitoring tool

**Answer:** B

---

### Q6. What does the control plane do in Kubernetes?

a) Run application containers
b) Make cluster-wide decisions and manage cluster state
c) Store data
d) Provide networking

**Answer:** B

---

## Section 2: Deployments and Scaling (5 questions)

### Q7. How do you scale a Deployment to 5 replicas?

a) `kubectl scale deployment myapp --replicas=5`
b) `kubectl resize myapp 5`
c) `kubectl grow myapp 5`
d) Delete and recreate

**Answer:** A

---

### Q8. What is a ReplicaSet?

a) A backup system
b) Maintains specified number of Pod replicas
c) A type of storage
d) A network configuration

**Answer:** B

---

### Q9. What is the Horizontal Pod Autoscaler (HPA)?

a) Manual scaling tool
b) Automatically scales Pods based on metrics
c) Vertical scaling only
d) Storage autoscaler

**Answer:** B

---

### Q10. Which metric does HPA commonly use for scaling?

a) Disk usage
b) CPU utilization
c) Number of files
d) Time of day

**Answer:** B

---

### Q11. What is a rolling update in Kubernetes?

a) Deleting all Pods at once
b) Gradually replacing old Pods with new ones
c) Backing up data
d) Rotating logs

**Answer:** B

---

## Section 3: Storage and Configuration (5 questions)

### Q12. What is a PersistentVolume (PV)?

a) Temporary storage
b) Cluster-wide storage resource
c) A Pod configuration
d) A network setting

**Answer:** B

---

### Q13. What is a PersistentVolumeClaim (PVC)?

a) A storage provider
b) Request for storage by a Pod
c) A backup system
d) A monitoring tool

**Answer:** B

---

### Q14. What is the purpose of ConfigMaps?

a) Store secrets
b) Store non-sensitive configuration data
c) Manage networks
d) Scale Pods

**Answer:** B

---

### Q15. What should you use to store sensitive data like API keys?

a) ConfigMap
b) Secret
c) Environment variables only
d) Plain text files

**Answer:** B

---

### Q16. What is the difference between ConfigMap and Secret?

a) No difference
b) Secrets are base64 encoded and have access controls
c) ConfigMaps are faster
d) Secrets are deprecated

**Answer:** B

---

## Section 4: Networking (4 questions)

### Q17. What type of Service exposes Pods to the internet?

a) ClusterIP
b) LoadBalancer
c) NodePort
d) Internal

**Answer:** B

---

### Q18. What is an Ingress in Kubernetes?

a) Data input system
b) HTTP/HTTPS routing to services
c) Storage access
d) Pod creation tool

**Answer:** B

---

### Q19. What is the default Service type in Kubernetes?

a) LoadBalancer
b) ClusterIP
c) NodePort
d) ExternalName

**Answer:** B

---

### Q20. How do Pods in the same cluster communicate by default?

a) They cannot
b) Using cluster-internal DNS and IP addresses
c) Only through external load balancers
d) Via the internet

**Answer:** B

---

## Section 5: ML-Specific Kubernetes (10 questions)

### Q21. How do you enable GPU access for a Pod?

a) Not possible
b) Specify GPU in resources.limits
c) GPUs work automatically
d) Use a special base image

**Answer:** B

---

### Q22. What is the NVIDIA device plugin for Kubernetes?

a) A monitoring tool
b) Enables GPU scheduling in Kubernetes
c) A container runtime
d) A storage driver

**Answer:** B

---

### Q23. What is a Job in Kubernetes?

a) A continuous service
b) Run-to-completion task, ideal for training jobs
c) A storage type
d) A network policy

**Answer:** B

---

### Q24. What is the difference between a Job and a Deployment?

a) No difference
b) Jobs run to completion, Deployments run continuously
c) Jobs are faster
d) Deployments are deprecated

**Answer:** B

---

### Q25. What is Kubeflow?

a) A cloud provider
b) ML platform built on Kubernetes
c) A container registry
d) A monitoring tool

**Answer:** B

---

### Q26. Why use Init Containers for ML workloads?

a) Increase speed
b) Download models/data before main container starts
c) Monitor training
d) No real use

**Answer:** B

---

### Q27. What is a StatefulSet used for?

a) Stateless applications
b) Applications requiring stable identity and storage
c) Temporary jobs
d) Configuration only

**Answer:** B

---

### Q28. When should you use a StatefulSet instead of Deployment for ML?

a) Never
b) For distributed training requiring stable network identities
c) Always
d) Only for inference

**Answer:** B

---

### Q29. What is the purpose of resource requests in Kubernetes?

a) Request data
b) Specify minimum resources Pod needs
c) Request more nodes
d) Send HTTP requests

**Answer:** B

---

### Q30. What happens if a Pod exceeds its memory limit?

a) Nothing
b) Pod is terminated (OOMKilled)
c) Automatically gets more memory
d) Cluster shuts down

**Answer:** B

---

## Answer Key

1. B   2. B   3. B   4. B   5. B
6. B   7. A   8. B   9. B   10. B
11. B  12. B  13. B  14. B  15. B
16. B  17. B  18. B  19. B  20. B
21. B  22. B  23. B  24. B  25. B
26. B  27. B  28. B  29. B  30. B

---

## Scoring

- **27-30 correct (90-100%)**: Excellent! Kubernetes expert
- **24-26 correct (80-89%)**: Good! Ready for advanced topics
- **21-23 correct (70-79%)**: Fair. Review weak areas
- **Below 21 (< 70%)**: Review module materials

---

## Next Module:** Module 05 - Data Pipelines or Module 06 - MLOps
