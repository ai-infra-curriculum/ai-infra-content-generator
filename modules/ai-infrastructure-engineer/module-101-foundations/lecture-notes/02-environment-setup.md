# Lesson 02: Development Environment Setup

**Duration:** 5 hours

## Learning Objectives

By the end of this lecture, you will be able to:
- Install and configure Python 3.9+ with virtual environment management
- Set up Git for version control and configure basic workflows
- Install Docker and understand basic containerization concepts
- Install Kubernetes CLI tools (kubectl) and verify cluster connectivity
- Configure cloud platform CLI tools (AWS, GCP, or Azure)
- Set up a professional code editor (VS Code or PyCharm) with ML extensions
- Create and manage development environments for ML infrastructure projects

## Overview

A well-configured development environment is crucial for efficient ML infrastructure work. In this lesson, you'll install and configure:

- Python 3.9+ with virtual environments
- Git for version control
- Docker for containerization
- Kubernetes command-line tools
- Cloud platform CLI (AWS/GCP/Azure)
- Code editor (VS Code or PyCharm)
- Essential Python packages

By the end, you'll have a professional ML infrastructure development environment.

## Prerequisites

- Computer with 8GB+ RAM (16GB recommended)
- 20GB+ free disk space
- Admin/sudo privileges
- Stable internet connection
- Linux, macOS, or Windows with WSL2

## Step 1: Operating System Preparation

### Linux (Ubuntu/Debian recommended)

If you're on Linux, you're all set! Most tools work natively.

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential curl wget git
```

### macOS

macOS works great for development. Install Homebrew first:

```bash
# Install Homebrew (package manager for macOS)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify installation
brew --version
```

### Windows with WSL2

For Windows, use Windows Subsystem for Linux 2 (WSL2):

```powershell
# In PowerShell (Administrator)
wsl --install

# Restart computer

# After restart, set Ubuntu as default
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

Once installed, open Ubuntu from Start menu and follow Linux instructions.

## Step 2: Install Python 3.9+

### Check Current Python Version

```bash
python3 --version
```

If you have Python 3.9 or higher, skip to virtual environment setup.

### Install Python 3.11 (Recommended)

**Ubuntu/Debian:**
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

**macOS:**
```bash
brew install python@3.11
```

### Set Up Python Virtual Environment

Virtual environments isolate project dependencies:

```bash
# Install pip if needed
python3.11 -m ensurepip --upgrade

# Create virtual environment for this course
python3.11 -m venv ~/ai-infra-venv

# Activate virtual environment
source ~/ai-infra-venv/bin/activate

# On Windows WSL:
source ~/ai-infra-venv/bin/activate

# Verify Python in venv
which python
python --version  # Should show 3.11.x
```

**Add activation to your shell profile** for convenience:

```bash
# For bash
echo 'alias activate-ai="source ~/ai-infra-venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc

# For zsh (macOS default)
echo 'alias activate-ai="source ~/ai-infra-venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
```

Now you can just type `activate-ai` to activate the environment.

### Install Essential Python Packages

```bash
# Make sure venv is activated
source ~/ai-infra-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core ML and infrastructure packages
pip install \
    torch torchvision \
    tensorflow \
    transformers \
    numpy pandas scikit-learn \
    fastapi uvicorn \
    requests httpx \
    pyyaml python-dotenv \
    pytest pytest-asyncio \
    black flake8 mypy
```

This will take 5-10 minutes. Grab a coffee!

## Step 3: Install Git

Git is essential for version control.

**Ubuntu/Debian:**
```bash
sudo apt install -y git
```

**macOS:**
```bash
brew install git
```

**Configure Git:**
```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### GitHub Account Setup

1. Create account at [github.com](https://github.com) if you don't have one
2. Set up SSH key for authentication:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Press Enter to accept default location
# Enter a passphrase (optional but recommended)

# Start SSH agent
eval "$(ssh-agent -s)"

# Add SSH key
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Copy the output
```

3. Go to GitHub → Settings → SSH and GPG keys → New SSH key
4. Paste the public key and save
5. Test connection:

```bash
ssh -T git@github.com
# Should see: "Hi username! You've successfully authenticated"
```

## Step 4: Install Docker

Docker is crucial for containerization.

### Ubuntu/Debian

```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# Log out and log back in for group change to take effect
```

### macOS

```bash
# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop from Applications
# Wait for it to finish starting (whale icon in menu bar)
```

### Windows (WSL2)

Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)

Enable WSL2 integration:
- Open Docker Desktop → Settings → Resources → WSL Integration
- Enable integration with Ubuntu

### Verify Docker Installation

```bash
# Check version
docker --version

# Run test container
docker run hello-world

# Should see "Hello from Docker!" message
```

## Step 5: Install Kubernetes Tools

### kubectl (Kubernetes CLI)

**Linux:**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**macOS:**
```bash
brew install kubectl
```

**Verify:**
```bash
kubectl version --client
```

### minikube (Local Kubernetes)

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**macOS:**
```bash
brew install minikube
```

**Verify:**
```bash
minikube version

# Start minikube (we'll use this in Module 04)
# Don't run this yet if low on resources
# minikube start --driver=docker
```

### Helm (Kubernetes Package Manager)

**Linux/macOS:**
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

**Verify:**
```bash
helm version
```

## Step 6: Install Cloud Platform CLI

Choose one cloud platform to start (you can add others later).

### AWS CLI

```bash
# Linux/macOS
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# macOS alternative
brew install awscli

# Verify
aws --version

# Configure (if you have AWS account)
aws configure
# Enter Access Key ID, Secret Access Key, region (us-east-1), output format (json)
```

### Google Cloud SDK

```bash
# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# macOS
brew install --cask google-cloud-sdk

# Initialize
gcloud init
```

### Azure CLI

```bash
# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# macOS
brew update && brew install azure-cli

# Login
az login
```

**For this course, having at least one cloud CLI is sufficient.**

## Step 7: Install Code Editor

### VS Code (Recommended)

**Linux:**
```bash
# Download .deb package
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code
```

**macOS:**
```bash
brew install --cask visual-studio-code
```

**Essential VS Code Extensions:**
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension eamodio.gitlens
code --install-extension redhat.vscode-yaml
code --install-extension github.copilot  # Optional, requires subscription
```

### PyCharm (Alternative)

Download PyCharm Community (free) from [jetbrains.com](https://www.jetbrains.com/pycharm/)

## Step 8: Verify Complete Setup

Run this verification script:

```bash
#!/bin/bash
# save as verify-setup.sh

echo "=== Verifying ML Infrastructure Development Environment ==="
echo ""

# Python
echo "Python Version:"
python --version || echo "❌ Python not found"
echo ""

# Pip packages
echo "Key Python Packages:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "❌ PyTorch not installed"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || echo "❌ TensorFlow not installed"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || echo "❌ FastAPI not installed"
echo ""

# Git
echo "Git Version:"
git --version || echo "❌ Git not found"
echo ""

# Docker
echo "Docker Version:"
docker --version || echo "❌ Docker not found"
docker ps > /dev/null 2>&1 && echo "✅ Docker running" || echo "⚠️  Docker not running (start Docker Desktop)"
echo ""

# Kubernetes
echo "Kubernetes Tools:"
kubectl version --client --short || echo "❌ kubectl not found"
minikube version --short || echo "❌ minikube not found"
helm version --short || echo "❌ helm not found"
echo ""

# Cloud CLI (at least one)
echo "Cloud CLIs:"
aws --version 2>/dev/null && echo "✅ AWS CLI installed" || echo "⚠️  AWS CLI not installed"
gcloud --version 2>/dev/null && echo "✅ GCloud CLI installed" || echo "⚠️  GCloud CLI not installed"
az --version 2>/dev/null && echo "✅ Azure CLI installed" || echo "⚠️  Azure CLI not installed"
echo ""

echo "=== Verification Complete ==="
echo "If you see ❌ errors, review the installation steps for those tools."
```

Run it:
```bash
chmod +x verify-setup.sh
./verify-setup.sh
```

## Step 9: Create Project Structure

Set up a workspace for this course:

```bash
# Create workspace directory
mkdir -p ~/ai-infra-learning
cd ~/ai-infra-learning

# Initialize git repository
git init

# Create standard structure
mkdir -p {modules/01-foundations,projects/{project-01,project-02,project-03},exercises,notes}

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
.pytest_cache/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment variables
.env
.env.local

# Data files (don't commit large files)
*.csv
*.h5
*.pkl
*.pth
*.ckpt

# Model files
models/
*.model
*.weights

# Docker
*.tar
EOF

# Create README
cat > README.md << 'EOF'
# AI Infrastructure Learning Journey

This repository contains my work for the AI Infrastructure Engineer curriculum.

## Structure

- `modules/` - Notes and exercises for each module
- `projects/` - Hands-on projects
- `exercises/` - Practice exercises
- `notes/` - Personal learning notes

## Progress

- [ ] Module 01: Foundations
- [ ] Project 01: Basic Model Serving
- ...
EOF

# Initial commit
git add .
git commit -m "Initial setup: Project structure"

# Connect to GitHub (create repo at github.com first)
# git remote add origin git@github.com:yourusername/ai-infra-learning.git
# git push -u origin main
```

## Troubleshooting

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Python Package Installation Fails

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Try again
pip install <package-name>
```

### minikube Won't Start

```bash
# Delete and recreate
minikube delete
minikube start --driver=docker --memory=4096
```

### Cloud CLI Authentication Issues

```bash
# AWS
aws configure

# GCloud
gcloud auth login
gcloud config set project <your-project-id>

# Azure
az login
```

## Next Steps

Congratulations! You now have a professional ML infrastructure development environment.

**In Lesson 03**, you'll dive deep into ML infrastructure concepts and the ML lifecycle.

But first, complete Exercise 01 to practice with your new environment.

---

## Exercise 01: Environment Verification

**File:** `exercises/exercise-01-environment.md`

**Tasks:**
1. Run the verification script and ensure all tools are installed
2. Create a simple Python script that imports PyTorch and TensorFlow
3. Run a Docker container and verify it works
4. Create a git repository and make your first commit
5. Take screenshots of successful execution

**Deliverable:**
- GitHub repository with code and screenshots
- README documenting your setup process

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)

---

**Next Lesson:** [03-ml-infrastructure-basics.md](./03-ml-infrastructure-basics.md) - Understanding the ML lifecycle and infrastructure requirements
