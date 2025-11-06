# Exercise 05: ML Framework Benchmarking Tool

## Learning Objectives

By completing this exercise, you will:
- Compare performance across PyTorch, TensorFlow, and JAX
- Benchmark GPU vs CPU training performance
- Profile memory usage and optimization
- Understand framework trade-offs for production
- Build automated benchmarking infrastructure

## Overview

Choosing the right ML framework is critical for AI infrastructure. This exercise builds a comprehensive benchmarking tool that helps teams make data-driven framework decisions based on their specific use cases.

## Prerequisites

- Python 3.11+ with PyTorch, TensorFlow, JAX installed
- Understanding of basic neural networks
- GPU access (optional but recommended for full benchmarking)
- Knowledge of Python profiling tools

## Problem Statement

Build `mlbench`, a framework benchmarking tool that:

1. **Trains identical models** across PyTorch, TensorFlow, and JAX
2. **Measures performance metrics**: training time, throughput, memory usage
3. **Compares GPU vs CPU** performance
4. **Profiles memory** usage and identifies bottlenecks
5. **Generates comprehensive reports** with visualizations

## Requirements

### Functional Requirements

#### FR1: Model Implementations
- Implement identical models in PyTorch, TensorFlow, JAX:
  - CNN for image classification (ResNet-18/34)
  - Transformer for NLP (BERT-like)
  - MLP for tabular data
- Ensure architectural parity across frameworks
- Use same hyperparameters (learning rate, batch size, etc.)

#### FR2: Benchmark Metrics
- Training time (per epoch, total)
- Throughput (samples/second)
- GPU utilization (%)
- Memory usage (peak, average)
- Model size (parameters, disk size)
- Inference latency (mean, p50, p95, p99)

#### FR3: Device Comparison
- CPU vs GPU training
- Single GPU vs multi-GPU
- Mixed precision (FP32 vs FP16)
- Batch size impact

#### FR4: Reporting
- Generate JSON/CSV results
- Create visualizations (charts, graphs)
- HTML report with comparisons
- Recommendations based on results

### Non-Functional Requirements

#### NFR1: Accuracy
- Ensure models train to similar accuracy (±2%)
- Validate correctness before benchmarking
- Use fixed random seeds for reproducibility

#### NFR2: Automation
- Run all benchmarks with single command
- Support partial benchmarks (skip frameworks)
- Resume interrupted benchmarks
- Progress tracking and ETA

#### NFR3: Extensibility
- Easy to add new frameworks
- Pluggable model architectures
- Configurable via YAML/JSON

## Implementation Tasks

### Task 1: Framework Abstraction Layer (4-5 hours)

Create a unified interface for all frameworks:

```python
# src/framework_interface.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import time

@dataclass
class BenchmarkResult:
    """Benchmark results for a single run"""
    framework: str
    model_name: str
    device: str  # "cpu", "cuda:0", "mps"
    batch_size: int
    precision: str  # "fp32", "fp16"

    # Training metrics
    train_time_per_epoch: float  # seconds
    total_train_time: float
    samples_per_second: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float

    # Accuracy metrics
    final_train_accuracy: float
    final_val_accuracy: float

    # Model metrics
    num_parameters: int
    model_size_mb: float

    # Inference metrics
    inference_latency_mean_ms: float
    inference_latency_p95_ms: float
    inference_latency_p99_ms: float

class FrameworkInterface(ABC):
    """Abstract interface for ML frameworks"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def build_model(self, model_type: str, **kwargs) -> Any:
        """
        Build model of specified type

        TODO: Implement for each framework
        - "cnn": ResNet-18 for image classification
        - "transformer": BERT-like for NLP
        - "mlp": Multi-layer perceptron

        Args:
            model_type: Type of model to build
            **kwargs: Model-specific parameters

        Returns:
            Model object
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        model: Any,
        train_loader: Any,
        optimizer: Any,
        loss_fn: Any
    ) -> Dict[str, float]:
        """
        Train for one epoch

        TODO: Implement training loop
        - Forward pass
        - Backward pass
        - Optimizer step
        - Track metrics

        Returns:
            {"loss": float, "accuracy": float, "time": float}
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        val_loader: Any,
        loss_fn: Any
    ) -> Dict[str, float]:
        """
        Evaluate model

        TODO: Implement evaluation loop
        Returns: {"loss": float, "accuracy": float}
        """
        pass

    @abstractmethod
    def benchmark_inference(
        self,
        model: Any,
        input_shape: Tuple,
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        Benchmark inference latency

        TODO:
        1. Warmup (10 runs)
        2. Run inference num_runs times
        3. Measure latency for each run
        4. Calculate statistics (mean, p50, p95, p99)

        Returns:
            {
                "mean_ms": float,
                "p50_ms": float,
                "p95_ms": float,
                "p99_ms": float
            }
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage

        TODO: Query framework-specific memory stats
        Returns: {"allocated_mb": float, "reserved_mb": float}
        """
        pass

    @abstractmethod
    def count_parameters(self, model: Any) -> int:
        """TODO: Count trainable parameters"""
        pass

    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """TODO: Save model to disk"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> Any:
        """TODO: Load model from disk"""
        pass
```

**Acceptance Criteria**:
- [ ] Abstract interface covers all frameworks
- [ ] Consistent API across implementations
- [ ] Memory tracking works on CPU and GPU
- [ ] Inference benchmarking is accurate

---

### Task 2: PyTorch Implementation (3-4 hours)

```python
# src/pytorch_impl.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import numpy as np
from typing import Any, Dict, Tuple

class PyTorchFramework(FrameworkInterface):
    """PyTorch implementation"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.torch_device = torch.device(device)

    def build_model(self, model_type: str, **kwargs) -> nn.Module:
        """
        TODO: Build PyTorch models

        CNN (ResNet-18):
        ```python
        model = models.resnet18(num_classes=kwargs.get("num_classes", 10))
        model = model.to(self.torch_device)
        return model
        ```

        Transformer:
        ```python
        from torch.nn import Transformer
        # Build BERT-like encoder
        ```

        MLP:
        ```python
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim):
                # Build multi-layer perceptron
        ```
        """
        pass

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: Any,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        TODO: PyTorch training loop

        ```python
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.torch_device), target.to(self.torch_device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        epoch_time = time.time() - start_time
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "time": epoch_time
        }
        ```
        """
        pass

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """TODO: PyTorch evaluation loop"""
        pass

    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple,
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        TODO: Benchmark PyTorch inference

        ```python
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(self.torch_device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.torch_device.type == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(dummy_input)

                if self.torch_device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
        ```
        """
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        TODO: Get PyTorch memory usage

        ```python
        if self.torch_device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.torch_device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.torch_device) / 1024**2
        else:
            # Use psutil for CPU memory
            import psutil
            process = psutil.Process()
            allocated = process.memory_info().rss / 1024**2
            reserved = allocated

        return {"allocated_mb": allocated, "reserved_mb": reserved}
        ```
        """
        pass

    def count_parameters(self, model: nn.Module) -> int:
        """
        TODO: Count PyTorch parameters

        ```python
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] All models build correctly
- [ ] Training loop works on CPU and GPU
- [ ] Memory tracking accurate
- [ ] Inference benchmarking reliable

---

### Task 3: TensorFlow Implementation (3-4 hours)

```python
# src/tensorflow_impl.py

import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from typing import Any, Dict, Tuple

class TensorFlowFramework(FrameworkInterface):
    """TensorFlow implementation"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        # Set device
        if device == "cpu":
            tf.config.set_visible_devices([], 'GPU')

    def build_model(self, model_type: str, **kwargs) -> keras.Model:
        """
        TODO: Build TensorFlow models

        CNN (ResNet-18):
        ```python
        model = keras.applications.ResNet50(
            weights=None,
            input_shape=kwargs.get("input_shape", (224, 224, 3)),
            classes=kwargs.get("num_classes", 10)
        )
        return model
        ```

        Transformer:
        ```python
        from tensorflow.keras.layers import (
            MultiHeadAttention, LayerNormalization, Dense
        )
        # Build transformer encoder
        ```

        MLP:
        ```python
        model = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(output_dim, activation='softmax')
        ])
        ```
        """
        pass

    def train_epoch(
        self,
        model: keras.Model,
        train_loader: Any,
        optimizer: keras.optimizers.Optimizer,
        loss_fn: keras.losses.Loss
    ) -> Dict[str, float]:
        """
        TODO: TensorFlow training loop

        ```python
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy()

        start_time = time.time()

        for data, target in train_loader:
            with tf.GradientTape() as tape:
                predictions = model(data, training=True)
                loss = loss_fn(target, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(target, predictions)

        epoch_time = time.time() - start_time

        return {
            "loss": float(train_loss.result()),
            "accuracy": float(train_accuracy.result() * 100),
            "time": epoch_time
        }
        ```
        """
        pass

    def evaluate(
        self,
        model: keras.Model,
        val_loader: Any,
        loss_fn: keras.losses.Loss
    ) -> Dict[str, float]:
        """TODO: TensorFlow evaluation"""
        pass

    def benchmark_inference(
        self,
        model: keras.Model,
        input_shape: Tuple,
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        TODO: Benchmark TensorFlow inference

        Similar to PyTorch but use TensorFlow APIs
        - Use tf.function for compiled inference
        - Use tf.timestamp() for accurate timing
        """
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        TODO: Get TensorFlow memory usage

        ```python
        if tf.config.list_physical_devices('GPU'):
            # GPU memory
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            allocated = memory_info['current'] / 1024**2
            reserved = memory_info['peak'] / 1024**2
        else:
            # CPU memory
            import psutil
            process = psutil.Process()
            allocated = process.memory_info().rss / 1024**2
            reserved = allocated

        return {"allocated_mb": allocated, "reserved_mb": reserved}
        ```
        """
        pass

    def count_parameters(self, model: keras.Model) -> int:
        """
        TODO: Count TensorFlow parameters

        ```python
        return int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] All TensorFlow models build correctly
- [ ] Training matches PyTorch performance (±5%)
- [ ] Memory tracking works
- [ ] Inference benchmarking accurate

---

### Task 4: JAX Implementation (4-5 hours)

```python
# src/jax_impl.py

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import time
import numpy as np
from typing import Any, Dict, Tuple

class JAXFramework(FrameworkInterface):
    """JAX/Flax implementation"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        # JAX device management
        if device == "cpu":
            jax.config.update('jax_platform_name', 'cpu')

    def build_model(self, model_type: str, **kwargs) -> nn.Module:
        """
        TODO: Build JAX/Flax models

        CNN:
        ```python
        class ResNet18(nn.Module):
            num_classes: int = 10

            @nn.compact
            def __call__(self, x, training: bool = False):
                # Build ResNet-18 architecture
                # Use nn.Conv, nn.BatchNorm, nn.relu
                pass

        return ResNet18(num_classes=kwargs.get("num_classes", 10))
        ```

        MLP:
        ```python
        class MLP(nn.Module):
            hidden_dims: Tuple[int, ...]
            num_classes: int

            @nn.compact
            def __call__(self, x):
                for dim in self.hidden_dims:
                    x = nn.Dense(dim)(x)
                    x = nn.relu(x)
                x = nn.Dense(self.num_classes)(x)
                return x
        ```
        """
        pass

    def train_epoch(
        self,
        model: nn.Module,
        params: Any,
        train_loader: Any,
        optimizer_state: Any,
        tx: optax.GradientTransformation
    ) -> Dict[str, float]:
        """
        TODO: JAX training loop

        ```python
        @jax.jit
        def train_step(params, opt_state, batch):
            def loss_fn(params):
                logits = model.apply({'params': params}, batch['image'])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits, batch['label']
                ).mean()
                return loss, logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = tx.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])

            return new_params, new_opt_state, loss, accuracy

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        start_time = time.time()

        for batch in train_loader:
            params, optimizer_state, loss, acc = train_step(
                params, optimizer_state, batch
            )
            total_loss += float(loss)
            total_accuracy += float(acc)
            num_batches += 1

        epoch_time = time.time() - start_time

        return {
            "loss": total_loss / num_batches,
            "accuracy": (total_accuracy / num_batches) * 100,
            "time": epoch_time
        }
        ```
        """
        pass

    def benchmark_inference(
        self,
        model: nn.Module,
        params: Any,
        input_shape: Tuple,
        num_runs: int = 1000
    ) -> Dict[str, float]:
        """
        TODO: Benchmark JAX inference

        ```python
        # Create JIT-compiled inference function
        @jax.jit
        def inference_fn(params, x):
            return model.apply({'params': params}, x)

        dummy_input = jnp.ones((1, *input_shape))

        # Warmup
        for _ in range(10):
            _ = inference_fn(params, dummy_input)
            jax.block_until_ready(_)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = inference_fn(params, dummy_input)
            jax.block_until_ready(output)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies = np.array(latencies)
        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
        ```
        """
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        TODO: Get JAX memory usage

        JAX doesn't expose memory APIs directly,
        use nvidia-smi or psutil
        """
        pass
```

**Acceptance Criteria**:
- [ ] JAX models build and train
- [ ] JIT compilation works correctly
- [ ] Performance competitive with PyTorch
- [ ] Inference benchmarking accurate

---

### Task 5: Benchmarking Orchestrator (5-6 hours)

```python
# src/benchmark_runner.py

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
import json
import yaml
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    frameworks: List[str]  # ["pytorch", "tensorflow", "jax"]
    models: List[str]  # ["cnn", "transformer", "mlp"]
    devices: List[str]  # ["cpu", "cuda:0"]
    batch_sizes: List[int]  # [32, 64, 128]
    num_epochs: int = 5
    dataset: str = "cifar10"
    precision: List[str] = None  # ["fp32", "fp16"]

    def __post_init__(self):
        if self.precision is None:
            self.precision = ["fp32"]

class BenchmarkRunner:
    """Run comprehensive benchmarks"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.console = Console()

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """
        TODO: Run all benchmark combinations

        ```python
        total_benchmarks = (
            len(self.config.frameworks) *
            len(self.config.models) *
            len(self.config.devices) *
            len(self.config.batch_sizes) *
            len(self.config.precision)
        )

        with Progress() as progress:
            task = progress.add_task(
                "[green]Running benchmarks...",
                total=total_benchmarks
            )

            for framework in self.config.frameworks:
                for model in self.config.models:
                    for device in self.config.devices:
                        for batch_size in self.config.batch_sizes:
                            for precision in self.config.precision:
                                result = self.run_single_benchmark(
                                    framework, model, device,
                                    batch_size, precision
                                )
                                self.results.append(result)
                                progress.update(task, advance=1)

        return self.results
        ```
        """
        pass

    def run_single_benchmark(
        self,
        framework: str,
        model: str,
        device: str,
        batch_size: int,
        precision: str
    ) -> BenchmarkResult:
        """
        TODO: Run single benchmark

        Steps:
        1. Initialize framework implementation
        2. Build model
        3. Load dataset
        4. Train for N epochs
        5. Measure metrics
        6. Benchmark inference
        7. Return results
        """
        pass

    def save_results(self, output_dir: Path) -> None:
        """
        TODO: Save results to disk

        Save as:
        - results.json (raw data)
        - results.csv (table format)
        - report.html (formatted report)
        """
        pass

    def print_summary(self) -> None:
        """
        TODO: Print results summary table

        ```python
        table = Table(title="Benchmark Results Summary")
        table.add_column("Framework", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Device", style="green")
        table.add_column("Batch Size", justify="right")
        table.add_column("Train Time (s)", justify="right")
        table.add_column("Throughput (samples/s)", justify="right")
        table.add_column("Peak Memory (MB)", justify="right")
        table.add_column("Inference (ms)", justify="right")

        for result in self.results:
            table.add_row(
                result.framework,
                result.model_name,
                result.device,
                str(result.batch_size),
                f"{result.total_train_time:.2f}",
                f"{result.samples_per_second:.0f}",
                f"{result.peak_memory_mb:.0f}",
                f"{result.inference_latency_mean_ms:.2f}"
            )

        self.console.print(table)
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] Runs all benchmark combinations
- [ ] Tracks progress with rich
- [ ] Saves results in multiple formats
- [ ] Generates summary tables

---

### Task 6: Visualization and Reporting (4-5 hours)

```python
# src/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
import pandas as pd

class BenchmarkVisualizer:
    """Generate visualizations from benchmark results"""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.df = pd.DataFrame([asdict(r) for r in results])

    def plot_training_time_comparison(self, output: Path) -> None:
        """
        TODO: Plot training time comparison

        ```python
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=self.df,
            x="model_name",
            y="total_train_time",
            hue="framework",
            ax=ax
        )

        ax.set_title("Training Time Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Time (seconds)")
        ax.legend(title="Framework")

        plt.tight_layout()
        plt.savefig(output / "training_time.png", dpi=300)
        plt.close()
        ```
        """
        pass

    def plot_throughput_comparison(self, output: Path) -> None:
        """TODO: Plot samples/second comparison"""
        pass

    def plot_memory_usage(self, output: Path) -> None:
        """TODO: Plot peak memory usage"""
        pass

    def plot_inference_latency(self, output: Path) -> None:
        """TODO: Plot inference latency comparison"""
        pass

    def plot_gpu_vs_cpu(self, output: Path) -> None:
        """
        TODO: Plot GPU vs CPU speedup

        Calculate speedup factor for each framework/model
        """
        pass

    def plot_batch_size_impact(self, output: Path) -> None:
        """TODO: Plot how batch size affects throughput"""
        pass

    def generate_html_report(self, output: Path) -> None:
        """
        TODO: Generate comprehensive HTML report

        ```python
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Framework Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .recommendation {{ background: #e7f3fe; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>ML Framework Benchmark Report</h1>

            <h2>Configuration</h2>
            <ul>
                <li>Frameworks: {", ".join(self.df['framework'].unique())}</li>
                <li>Models: {", ".join(self.df['model_name'].unique())}</li>
                <li>Devices: {", ".join(self.df['device'].unique())}</li>
            </ul>

            <h2>Results Summary</h2>
            {self.df.to_html()}

            <h2>Visualizations</h2>
            <img src="training_time.png" alt="Training Time">
            <img src="throughput.png" alt="Throughput">
            <img src="memory.png" alt="Memory Usage">
            <img src="inference.png" alt="Inference Latency">

            <h2>Recommendations</h2>
            <div class="recommendation">
                {self._generate_recommendations()}
            </div>
        </body>
        </html>
        '''

        (output / "report.html").write_text(html)
        ```
        """
        pass

    def _generate_recommendations(self) -> str:
        """
        TODO: Generate framework recommendations

        Analyze results and provide guidance:
        - Best for training speed
        - Best for inference latency
        - Most memory efficient
        - Best GPU utilization
        """
        pass
```

**Acceptance Criteria**:
- [ ] All visualizations generate correctly
- [ ] HTML report is comprehensive
- [ ] Recommendations are data-driven
- [ ] Charts are publication-quality

---

## Deliverables

1. **Source Code** (`src/`)
   - `framework_interface.py`
   - `pytorch_impl.py`
   - `tensorflow_impl.py`
   - `jax_impl.py`
   - `benchmark_runner.py`
   - `visualizer.py`
   - `cli.py`

2. **Configuration** (`configs/`)
   - `benchmark_config.yaml` - Default configuration
   - `quick_test.yaml` - Fast benchmark for testing
   - `comprehensive.yaml` - Full benchmark suite

3. **Tests** (`tests/`)
   - Unit tests for each framework
   - Integration tests for benchmarking
   - Mock tests for expensive operations

4. **Documentation**
   - README.md with usage examples
   - RESULTS.md with sample benchmark results
   - ANALYSIS.md with framework comparison analysis

---

## CLI Interface

```bash
# Run all benchmarks
mlbench run --config configs/benchmark_config.yaml

# Run specific framework
mlbench run --frameworks pytorch tensorflow

# Run specific model
mlbench run --models cnn --devices cuda:0

# Quick test (1 epoch)
mlbench run --config configs/quick_test.yaml

# Generate report from existing results
mlbench report --results results/benchmark_results.json --output report/

# Compare two benchmark runs
mlbench compare results1.json results2.json
```

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 30% | Models train to similar accuracy |
| **Completeness** | 25% | All frameworks implemented |
| **Benchmarking** | 20% | Accurate, reproducible measurements |
| **Reporting** | 15% | Clear visualizations and recommendations |
| **Code Quality** | 10% | Clean, well-tested code |

**Passing Score**: 70%
**Excellence**: 90%+

---

## Resources

- [PyTorch Benchmarking](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler)
- [JAX Benchmarking](https://jax.readthedocs.io/en/latest/async_dispatch.html)

---

## Estimated Time

- **Task 1** (Interface): 4-5 hours
- **Task 2** (PyTorch): 3-4 hours
- **Task 3** (TensorFlow): 3-4 hours
- **Task 4** (JAX): 4-5 hours
- **Task 5** (Orchestrator): 5-6 hours
- **Task 6** (Visualization): 4-5 hours
- **Testing**: 6-8 hours
- **Documentation**: 2-3 hours

**Total**: 31-40 hours

---

**This exercise provides hands-on experience with production ML infrastructure decisions and benchmarking methodologies.**
