# Sample Module: Observability Fundamentals

## Learning Objective

Explain core observability concepts and instrument a simple service to emit logs, metrics, and traces.

## Overview

Observability enables engineers to understand the internal state of complex systems. In this sample module you will:

- Differentiate between monitoring and observability.
- Configure log, metric, and trace emission for a Python service.
- Interpret signals to identify the root cause of a latency incident.

## Key Concepts

### The Three Pillars
- **Logs**: Discrete records of events or state changes.
- **Metrics**: Aggregated time-series measurements (e.g., request latency).
- **Traces**: End-to-end request flows with spans showing timing relationships.

### Telemetry Pipeline
1. Instrument code with an observability SDK (OpenTelemetry in this example).
2. Export telemetry to a collector or SaaS backend.
3. Visualize dashboards and set SLO-based alerts.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="https://otel.example.com/v1/traces"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

def handle_request():
    with tracer.start_as_current_span("handle_request"):
        # business logic
        ...
```

## Mini Case Study

**Scenario**: An inference API experiences periodic latency spikes.

1. Metrics show p95 latency rising from 120ms to 400ms.
2. Traces reveal a new cache lookup span with 300ms wait time.
3. Logs indicate cache connection pool exhaustion.

**Resolution**: Increase pool size, add backpressure, and tighten alert thresholds to detect regressions earlier.
