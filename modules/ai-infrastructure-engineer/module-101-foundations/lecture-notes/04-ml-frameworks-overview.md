# Lesson 04: ML Frameworks Overview

## Learning Objectives

- Understand popular ML frameworks (PyTorch, TensorFlow, scikit-learn)
- Choose appropriate framework for different use cases
- Install and configure ML frameworks
- Deploy models from different frameworks

## Duration: 2-3 hours

---

## 1. Popular ML Frameworks

### 1.1 PyTorch

**Overview**: Dynamic computation graph, research-friendly, Pythonic

**Strengths:**
- Intuitive Python API
- Excellent for research and prototyping
- Strong community and ecosystem
- Dynamic graphs (define-by-run)

**Use Cases:**
- Computer vision (ResNet, YOLO, Vision Transformers)
- NLP and LLMs (BERT, GPT, Llama)
- Research projects
- Custom model architectures

**Installation:**
```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Simple Example:**
```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# Inference
x = torch.randn(32, 784).to(device)  # batch_size=32
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

### 1.2 TensorFlow / Keras

**Overview**: Production-ready, static/dynamic graphs, comprehensive ecosystem

**Strengths:**
- Production deployment (TensorFlow Serving)
- TensorFlow Lite for mobile/edge
- TensorFlow.js for web
- Keras high-level API (easy to use)
- Strong enterprise adoption

**Use Cases:**
- Production ML systems
- Mobile and edge deployment
- Structured data (tabular)
- Time series forecasting

**Installation:**
```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA and cuDNN)
pip install tensorflow[and-cuda]
```

**Simple Example:**
```python
import tensorflow as tf
from tensorflow import keras

# Define model using Keras Sequential API
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Inference
import numpy as np
x = np.random.randn(32, 784)
predictions = model.predict(x)
print(predictions.shape)  # (32, 10)
```

### 1.3 Scikit-learn

**Overview**: Traditional ML, simple API, production-ready

**Strengths:**
- Easy to use and learn
- Comprehensive algorithms (SVM, Random Forest, XGBoost integration)
- Excellent documentation
- Fast prototyping

**Use Cases:**
- Classification, regression, clustering
- Feature engineering and preprocessing
- Baseline models
- Small to medium datasets

**Installation:**
```bash
pip install scikit-learn
```

**Simple Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

---

## 2. Framework Comparison

| Feature | PyTorch | TensorFlow | Scikit-learn |
|---------|---------|------------|--------------|
| **Learning Curve** | Medium | Medium-High | Easy |
| **Flexibility** | High | Medium | Low |
| **Production** | Good | Excellent | Excellent |
| **Mobile/Edge** | Limited | Excellent | N/A |
| **Community** | Large | Large | Large |
| **Use Case** | Research, DL | Production, DL | Traditional ML |
| **GPU Support** | Excellent | Excellent | Limited |

---

## 3. Model Deployment Frameworks

### 3.1 TorchServe (PyTorch)

```bash
# Install TorchServe
pip install torchserve torch-model-archiver

# Create model archive
torch-model-archiver --model-name resnet18 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet18.pth \
  --handler image_classifier

# Start server
torchserve --start --model-store model_store --models resnet18=resnet18.mar
```

### 3.2 TensorFlow Serving

```bash
# Docker deployment
docker pull tensorflow/serving

docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

### 3.3 ONNX (Cross-framework)

**ONNX** (Open Neural Network Exchange) enables model portability:

```python
# PyTorch to ONNX
import torch

model = torch.load('model.pth')
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")

# ONNX Runtime inference
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input.numpy()})
```

---

## 4. Hands-On Exercise: Framework Benchmarking

### Task: Compare PyTorch and TensorFlow inference speed

**TODO: Implement inference benchmarking for both frameworks**

```python
# TODO: Implement this benchmarking script
import time
import torch
import tensorflow as tf
import numpy as np

def benchmark_pytorch():
    """
    TODO: Create a simple PyTorch model and measure inference time
    - Create model with 2-3 layers
    - Run 1000 inferences
    - Calculate average time per inference
    """
    pass

def benchmark_tensorflow():
    """
    TODO: Create equivalent TensorFlow model and measure inference time
    - Create model with same architecture as PyTorch
    - Run 1000 inferences
    - Calculate average time per inference
    """
    pass

def main():
    # TODO: Run benchmarks and compare results
    # Print results in a formatted table
    pass

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Framework Benchmarking Results:
================================
PyTorch:     2.5ms per inference
TensorFlow:  2.3ms per inference
Winner:      TensorFlow (8% faster)
```

---

## 5. Key Takeaways

- ✅ PyTorch: Best for research and dynamic models
- ✅ TensorFlow: Best for production and mobile deployment
- ✅ Scikit-learn: Best for traditional ML and quick prototyping
- ✅ ONNX enables cross-framework model deployment
- ✅ Choose framework based on use case, not popularity

---

## 6. Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ONNX Documentation](https://onnx.ai/)

---

## Next Steps

✅ Complete benchmarking exercise
✅ Proceed to [Lesson 05: Cloud Platforms for ML](./05-cloud-platforms-intro.md)
