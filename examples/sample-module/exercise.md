# Exercise: Instrument a FastAPI Endpoint

## Goal

Add OpenTelemetry instrumentation to a FastAPI service and verify that traces and metrics are exported.

## Steps

1. **Clone the starter**
   ```bash
   git clone https://github.com/example/observability-starter.git
   cd observability-starter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Instrument the service**
   - Configure OTLP exporters for traces and metrics.
   - Wrap the `/predict` endpoint with a span.
   - Record a custom metric `inference.duration`.

4. **Run the service**
   ```bash
   uvicorn app:app --reload
   ```

5. **Generate traffic**
   ```bash
   hey -n 100 -c 5 http://localhost:8000/predict
   ```

6. **Verify telemetry**
   - Confirm spans appear in your collector backend.
   - Check that the `inference.duration` metric is emitted.

## Success Criteria

- Requests execute successfully.
- Traces show nested spans with latency breakdown.
- Custom metric reflects request latency in milliseconds.

## Stretch Goals

- Add log correlation (trace/span IDs in structured logs).
- Emit an alert when p95 latency exceeds 300ms.
