# Exercise 03: Kubernetes Deployment for ML

**Duration:** 3-4 hours
**Difficulty:** Advanced
**Prerequisites:** Lessons 07 & 08, Exercise 02 (Docker)

## Learning Objectives

By completing this exercise, you will:
- Deploy ML applications to Kubernetes
- Create and manage Deployments, Services, ConfigMaps
- Implement health checks and readiness probes
- Configure auto-scaling with HPA
- Manage application configuration
- Monitor deployed applications
- Perform rolling updates and rollbacks

---

## Prerequisites Setup

### Install Required Tools

**kubectl:**
```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify
kubectl version --client
```

**minikube (Local Kubernetes):**
```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start minikube
minikube start --cpus=2 --memory=4096

# Verify
kubectl get nodes
```

---

## Part 1: Basic Deployment (45 minutes)

### Task 1.1: Create a Simple Deployment

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api

spec:
  replicas: 2  # Start with 2 pods for HA

  selector:
    matchLabels:
      app: ml-api

  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-app:v2  # From Exercise 02
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
```

**Your Task:**
1. Load your Docker image into minikube:
```bash
# Build image if not already done
docker build -t ml-app:v2 .

# Load into minikube
minikube image load ml-app:v2
```

2. Apply the deployment:
```bash
kubectl apply -f deployment.yaml
```

3. Check deployment status:
```bash
kubectl get deployments
kubectl get pods
kubectl describe deployment ml-api
```

4. View pod logs:
```bash
# Get pod name
kubectl get pods

# View logs
kubectl logs <pod-name>

# Follow logs
kubectl logs -f <pod-name>
```

**Expected Results:**
- Deployment created successfully
- 2 pods running
- Pods show "Running" status

**Questions:**
1. What does `replicas: 2` mean?
2. Why do we set resource requests and limits?
3. What happens if you delete one pod?

---

### Task 1.2: Create a Service

Create `service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
  labels:
    app: ml-api

spec:
  type: LoadBalancer  # For minikube, use NodePort or LoadBalancer

  selector:
    app: ml-api  # Must match deployment labels

  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
```

**Your Task:**
1. Apply the service:
```bash
kubectl apply -f service.yaml
```

2. Get service details:
```bash
kubectl get services
kubectl describe service ml-api-service
```

3. Access the service:
```bash
# For minikube
minikube service ml-api-service --url

# Or use port forwarding
kubectl port-forward service/ml-api-service 8000:80

# Test
curl http://localhost:8000/health
```

**Expected Results:**
- Service created successfully
- Service has an endpoint
- Can access /health endpoint

---

## Part 2: Configuration Management (45 minutes)

### Task 2.1: Create a ConfigMap

Create `configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config

data:
  # Model configuration
  model_name: "resnet18"
  device: "cpu"

  # Server configuration
  port: "8000"
  log_level: "info"
  workers: "1"

  # Feature flags
  enable_metrics: "true"
  enable_cache: "false"

  # Inference settings
  default_top_k: "5"
  max_batch_size: "32"
```

**Your Task:**
1. Apply the ConfigMap:
```bash
kubectl apply -f configmap.yaml
```

2. View ConfigMap:
```bash
kubectl get configmap ml-api-config
kubectl describe configmap ml-api-config
```

3. Update deployment to use ConfigMap.

Modify `deployment.yaml`:
```yaml
spec:
  template:
    spec:
      containers:
      - name: ml-api
        # ... existing config ...

        # Add environment variables from ConfigMap
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: ml-api-config
              key: model_name
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: ml-api-config
              key: log_level
        - name: DEVICE
          valueFrom:
            configMapKeyRef:
              name: ml-api-config
              key: device

        # Or use envFrom to import all keys
        envFrom:
        - configMapRef:
            name: ml-api-config
```

4. Apply updated deployment:
```bash
kubectl apply -f deployment.yaml
```

5. Verify environment variables:
```bash
kubectl exec <pod-name> -- env | grep MODEL
```

**Expected Results:**
- ConfigMap created
- Pods have environment variables from ConfigMap
- Application reads configuration correctly

---

### Task 2.2: Update Configuration

**Your Task:**
1. Update ConfigMap (change `log_level` to `debug`):
```bash
kubectl edit configmap ml-api-config
# Or edit configmap.yaml and reapply
```

2. Restart pods to pick up new config:
```bash
kubectl rollout restart deployment ml-api
```

3. Verify new configuration:
```bash
kubectl exec <pod-name> -- env | grep LOG_LEVEL
```

**Questions:**
1. Why do we need to restart pods?
2. How would you automate configuration updates?

---

## Part 3: Health Checks and Probes (30 minutes)

### Task 3.1: Add Health Probes

Update `deployment.yaml` to add liveness and readiness probes:

```yaml
spec:
  template:
    spec:
      containers:
      - name: ml-api
        # ... existing config ...

        # Liveness probe: Restart pod if unhealthy
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60  # Wait for model to load
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness probe: Remove from service if not ready
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
```

**Your Task:**
1. Apply updated deployment
2. Watch pod status:
```bash
kubectl get pods -w
```

3. Check probe status:
```bash
kubectl describe pod <pod-name>
```

4. Test failure scenario:
   - Modify your app to return 500 from /health
   - Rebuild and reload image
   - Observe pod restarts

**Expected Results:**
- Pods show READY 1/1 when passing readiness
- Failed liveness probes trigger pod restarts
- Failed readiness probes remove pod from service

**Questions:**
1. What's the difference between liveness and readiness probes?
2. When should initialDelaySeconds be longer?

---

## Part 4: Autoscaling (45 minutes)

### Task 4.1: Install Metrics Server

```bash
# Install metrics server (required for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# For minikube, you may need to disable TLS
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Verify metrics server
kubectl top nodes
kubectl top pods
```

---

### Task 4.2: Create HPA

Create `hpa.yaml`:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa

spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api

  minReplicas: 2
  maxReplicas: 10

  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

**Your Task:**
1. Apply HPA:
```bash
kubectl apply -f hpa.yaml
```

2. Check HPA status:
```bash
kubectl get hpa
kubectl describe hpa ml-api-hpa
```

3. Generate load to test scaling:
```bash
# Get service URL
SERVICE_URL=$(minikube service ml-api-service --url)

# Install Apache Bench (if not installed)
# macOS: brew install apache-bench
# Linux: sudo apt-get install apache2-utils

# Generate load
ab -n 10000 -c 50 $SERVICE_URL/info
```

4. Watch scaling:
```bash
# Terminal 1: Watch HPA
kubectl get hpa -w

# Terminal 2: Watch pods
kubectl get pods -w

# Terminal 3: Watch metrics
watch kubectl top pods
```

**Expected Results:**
- Under load, CPU usage increases
- HPA scales up pods (2 → 4 → 6 → ...)
- When load stops, HPA scales down (after 5 minutes)

**Questions:**
1. How long does it take to scale up?
2. Why does scale-down take longer?
3. What happens if you hit maxReplicas?

---

## Part 5: Rolling Updates and Rollbacks (30 minutes)

### Task 5.1: Perform Rolling Update

Let's update to a new version of the app.

**Your Task:**
1. Modify `src/app.py` (change a message):
```python
@app.get("/")
def root():
    return {"message": "ML Model API v2.0", "model": "ResNet-18"}
```

2. Build new image:
```bash
docker build -t ml-app:v3 .
minikube image load ml-app:v3
```

3. Update deployment:
```bash
kubectl set image deployment/ml-api ml-api=ml-app:v3
```

4. Watch rollout:
```bash
kubectl rollout status deployment/ml-api

# See rollout history
kubectl rollout history deployment/ml-api
```

5. Verify update:
```bash
curl $(minikube service ml-api-service --url)/
```

**Expected Results:**
- Old pods terminated gradually
- New pods started one at a time
- No downtime during update
- New version running

---

### Task 5.2: Rollback Deployment

**Your Task:**
1. Check rollout history:
```bash
kubectl rollout history deployment/ml-api
```

2. Rollback to previous version:
```bash
kubectl rollout undo deployment/ml-api
```

3. Verify rollback:
```bash
kubectl rollout status deployment/ml-api
curl $(minikube service ml-api-service --url)/
```

4. Rollback to specific revision:
```bash
kubectl rollout undo deployment/ml-api --to-revision=1
```

**Expected Results:**
- Rollback completes successfully
- Previous version running
- No downtime

---

## Part 6: Monitoring and Debugging (45 minutes)

### Task 6.1: View Logs

**Your Task:**
1. View logs from all pods:
```bash
kubectl logs -l app=ml-api
```

2. Follow logs from specific pod:
```bash
kubectl logs -f <pod-name>
```

3. View previous container logs (if crashed):
```bash
kubectl logs <pod-name> --previous
```

4. Stream logs from multiple pods:
```bash
kubectl logs -f -l app=ml-api --all-containers=true
```

---

### Task 6.2: Debug Pods

**Your Task:**
1. Get pod details:
```bash
kubectl describe pod <pod-name>
```

2. Execute commands in pod:
```bash
# Interactive shell
kubectl exec -it <pod-name> -- /bin/bash

# Single command
kubectl exec <pod-name> -- env
kubectl exec <pod-name> -- ps aux
kubectl exec <pod-name> -- curl localhost:8000/health
```

3. Check events:
```bash
kubectl get events --sort-by='.lastTimestamp'
```

4. Port forward for debugging:
```bash
kubectl port-forward <pod-name> 8000:8000
```

---

### Task 6.3: Resource Usage

**Your Task:**
1. View current resource usage:
```bash
kubectl top nodes
kubectl top pods
kubectl top pods -l app=ml-api
```

2. Check resource requests vs limits:
```bash
kubectl describe nodes
```

3. Identify resource-heavy pods:
```bash
kubectl top pods --sort-by=cpu
kubectl top pods --sort-by=memory
```

---

## Part 7: Challenge Tasks (Optional, 1-2 hours)

### Challenge 1: Multi-Environment Setup

Create separate configurations for dev/staging/prod:

```bash
# Create namespaces
kubectl create namespace development
kubectl create namespace production

# Deploy to different namespaces
kubectl apply -f deployment.yaml -n development
kubectl apply -f deployment.yaml -n production

# Different configs per environment
kubectl apply -f configmap-dev.yaml -n development
kubectl apply -f configmap-prod.yaml -n production
```

### Challenge 2: Implement Blue-Green Deployment

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
      version: blue
  template:
    metadata:
      labels:
        app: ml-api
        version: blue
    spec:
      containers:
      - name: ml-api
        image: ml-app:v2

---
# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
      version: green
  template:
    metadata:
      labels:
        app: ml-api
        version: green
    spec:
      containers:
      - name: ml-api
        image: ml-app:v3

---
# Service (switch between blue and green)
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
    version: blue  # Change to 'green' to switch
  ports:
  - port: 80
    targetPort: 8000
```

### Challenge 3: Add PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: ml-api
```

This ensures at least 1 pod is always available during voluntary disruptions (node maintenance, cluster upgrades).

### Challenge 4: Implement Canary Deployment

Deploy new version to 10% of pods, gradually increase:

```yaml
# Stable deployment (90%)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-stable
spec:
  replicas: 9
  template:
    spec:
      containers:
      - name: ml-api
        image: ml-app:v2

---
# Canary deployment (10%)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-canary
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: ml-api
        image: ml-app:v3
```

---

## Verification Checklist

After completing this exercise, you should be able to:

- [ ] Deploy applications to Kubernetes
- [ ] Create and manage Deployments
- [ ] Create and manage Services (ClusterIP, NodePort, LoadBalancer)
- [ ] Use ConfigMaps for configuration
- [ ] Implement health checks (liveness, readiness)
- [ ] Set up auto-scaling with HPA
- [ ] Perform rolling updates
- [ ] Rollback failed deployments
- [ ] View and analyze logs
- [ ] Debug pod issues
- [ ] Monitor resource usage
- [ ] Understand pod lifecycle

---

## Cleanup

When done, clean up resources:

```bash
# Delete all resources
kubectl delete deployment ml-api
kubectl delete service ml-api-service
kubectl delete configmap ml-api-config
kubectl delete hpa ml-api-hpa

# Or delete everything in namespace
kubectl delete all --all

# Stop minikube
minikube stop

# Delete minikube cluster
minikube delete
```

---

## Submission (If part of course)

Create a ZIP file with:
1. All YAML files (deployment, service, configmap, hpa)
2. Screenshot of running pods
3. Screenshot of HPA in action
4. Screenshot of rolling update
5. Answers to questions throughout exercise
6. Output of `kubectl get all`

---

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kubernetes Patterns](https://k8spatterns.io/)
- [Production Best Practices](https://kubernetes.io/docs/setup/best-practices/)
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

---

**Next Steps:** Complete [Project 01](../../projects/project-101-basic-model-serving/) to build a complete ML serving system!
