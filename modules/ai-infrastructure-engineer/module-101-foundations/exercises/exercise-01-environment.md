# Exercise 01: Environment Verification

**Module:** 01 - Foundations
**Duration:** 2-3 hours
**Difficulty:** Beginner
**Objective:** Verify your development environment is properly configured

## Overview

This exercise ensures you have correctly installed and configured all tools needed for ML infrastructure development. You'll create simple programs to test each component.

## Prerequisites

- Completed Lesson 02: Environment Setup
- All tools installed as per lesson instructions

## Tasks

### Task 1: Python Environment Verification (30 minutes)

Create a Python script that verifies your ML packages.

**File:** `verify_python.py`

```python
#!/usr/bin/env python3
"""
Verification script for Python ML environment
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def main():
    print("=" * 50)
    print("Python ML Environment Verification")
    print("=" * 50)
    print(f"\nPython Version: {sys.version}")
    print(f"Python Path: {sys.executable}\n")

    # Core ML packages
    packages = [
        ('PyTorch', 'torch'),
        ('TensorFlow', 'tensorflow'),
        ('Transformers', 'transformers'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('FastAPI', 'fastapi'),
        ('Uvicorn', 'uvicorn'),
        ('Requests', 'requests'),
        ('HTTPX', 'httpx'),
        ('PyYAML', 'yaml'),
        ('python-dotenv', 'dotenv'),
        ('pytest', 'pytest'),
        ('Black', 'black'),
        ('Flake8', 'flake8'),
    ]

    results = []
    print("Checking packages...\n")
    for package, import_name in packages:
        results.append(check_package(package, import_name))

    # Summary
    print("\n" + "=" * 50)
    print(f"Summary: {sum(results)}/{len(results)} packages installed")
    print("=" * 50)

    if all(results):
        print("\n🎉 All packages installed successfully!")
        return 0
    else:
        print("\n⚠️  Some packages are missing. Please install them.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Run the script:**
```bash
# Activate virtual environment
source ~/ai-infra-venv/bin/activate

# Run verification
python verify_python.py
```

**Expected Output:**
- ✅ for all packages
- "All packages installed successfully!"

**Deliverable 1:**
- Screenshot or text output showing all packages installed

---

### Task 2: Docker Verification (30 minutes)

Create and run a simple Docker container.

**File:** `Dockerfile.test`

```dockerfile
# Simple test Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy verification script
COPY verify_python.py .

# Install packages
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy pandas scikit-learn

# Run verification
CMD ["python", "verify_python.py"]
```

**Build and run:**
```bash
# Build Docker image
docker build -f Dockerfile.test -t ml-env-test .

# Run container
docker run ml-env-test

# Check Docker is working
docker ps -a
docker images
```

**Deliverable 2:**
- Screenshot of successful Docker build
- Screenshot of container running

---

### Task 3: Git Repository Setup (30 minutes)

Create a Git repository for your learning journey.

```bash
# Create new repository
mkdir -p ~/ai-infra-learning
cd ~/ai-infra-learning

# Initialize git
git init

# Create project structure
mkdir -p modules/01-foundations exercises projects

# Copy your verification scripts
cp /path/to/verify_python.py exercises/

# Create README
cat > README.md << 'EOF'
# AI Infrastructure Learning Journey

My learning repository for AI Infrastructure Engineer curriculum.

## Environment

- Python 3.11
- Docker installed
- Kubernetes tools ready
- VS Code configured

## Progress

- [x] Module 01: Started
- [ ] Module 02: Not started
- [ ] Project 01: Not started
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
venv/
.env
*.log
.DS_Store
EOF

# First commit
git add .
git commit -m "Initial commit: Environment setup verification"

# Check status
git log
git status
```

**Create GitHub repository:**
1. Go to github.com
2. Click "New repository"
3. Name it "ai-infra-learning"
4. Do NOT initialize with README (you already have one)
5. Copy the remote URL

**Push to GitHub:**
```bash
git remote add origin git@github.com:yourusername/ai-infra-learning.git
git branch -M main
git push -u origin main
```

**Deliverable 3:**
- Link to your GitHub repository
- Screenshot showing your first commit

---

### Task 4: Kubernetes Verification (30 minutes)

Verify Kubernetes tools are working.

```bash
# Check kubectl
kubectl version --client

# Check minikube
minikube version

# Check helm
helm version

# Start minikube (if resources allow)
minikube start --driver=docker --memory=2048

# Verify cluster
kubectl cluster-info
kubectl get nodes

# Deploy test pod
kubectl run test-nginx --image=nginx --port=80

# Check pod
kubectl get pods
kubectl describe pod test-nginx

# Clean up
kubectl delete pod test-nginx
minikube stop
```

**Deliverable 4:**
- Screenshot of kubectl, minikube, helm versions
- Screenshot of test pod running

---

### Task 5: Cloud CLI Verification (30 minutes)

Verify at least one cloud CLI is working.

**AWS:**
```bash
# Check version
aws --version

# List regions (if configured)
aws ec2 describe-regions --output table
```

**GCP:**
```bash
# Check version
gcloud --version

# List projects
gcloud projects list
```

**Azure:**
```bash
# Check version
az --version

# List locations
az account list-locations --output table
```

**Deliverable 5:**
- Screenshot of cloud CLI working
- Note which cloud platform you're using

---

### Task 6: VS Code Configuration (20 minutes)

Configure VS Code for Python ML development.

**Tasks:**
1. Open VS Code
2. Install Python extension (if not already)
3. Open your `ai-infra-learning` folder
4. Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "~/ai-infra-venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

5. Create a test Python file and verify:
   - Syntax highlighting works
   - Autocomplete works
   - Linting works (create intentional error, see red squiggle)

**Deliverable 6:**
- Screenshot of VS Code with Python file open
- Screenshot showing autocomplete or linting working

---

## Challenge Tasks (Optional)

### Challenge 1: Simple ML Model Inference

Create a script that loads a pre-trained model and makes a prediction.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)
model.eval()

# Download a sample image
url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Prepare image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_batch)

# Get prediction
_, predicted = torch.max(output, 1)
print(f"Predicted class: {predicted.item()}")
```

### Challenge 2: Docker Compose Multi-Service

Create a `docker-compose.yml` with multiple services:

```yaml
version: '3.8'

services:
  app:
    image: python:3.11-slim
    volumes:
      - .:/app
    working_dir: /app
    command: sleep infinity

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

Run with: `docker-compose up -d`

---

## Submission

Create a document with:

1. **System Information**
   - OS and version
   - Hardware specs (CPU, RAM)

2. **All Deliverables**
   - Screenshots for each task
   - GitHub repository link
   - Notes on any issues encountered

3. **Reflection Questions**
   - What was the most challenging part of setup?
   - Which tool are you most excited to learn more about?
   - Any questions about the environment?

## Grading Criteria

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Python | 20 | All packages installed and verified |
| Task 2: Docker | 20 | Docker building and running containers |
| Task 3: Git | 20 | Repository created and pushed to GitHub |
| Task 4: Kubernetes | 15 | kubectl, minikube, helm working |
| Task 5: Cloud CLI | 10 | At least one cloud CLI functional |
| Task 6: VS Code | 10 | Editor configured for Python development |
| Documentation | 5 | Clear screenshots and explanations |
| **Total** | **100** | |

**Passing:** 70/100 points

## Common Issues and Solutions

### Python Packages Won't Install
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Install one at a time to identify problem package
pip install torch
pip install tensorflow
```

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### minikube Start Fails
```bash
# Use different driver
minikube start --driver=docker --memory=2048

# Or just verify minikube is installed
minikube version
```

### Git Push Authentication Failed
```bash
# Use SSH key instead
# Follow SSH setup in Lesson 02

# Or use personal access token for HTTPS
# Generate at github.com/settings/tokens
```

## Next Steps

Once you've completed this exercise and verified your environment:

1. **Proceed to Lesson 03** - ML Infrastructure Basics
2. **Keep your environment maintained** - Update packages regularly
3. **Practice daily** - Use these tools in your work or personal projects

---

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Docker Getting Started](https://docs.docker.com/get-started/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [Kubernetes Basics Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
