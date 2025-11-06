# Module 08: Monitoring and Observability - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (20/25 questions)
**Coverage:** Monitoring, logging, observability

---

## Section 1: Observability Fundamentals (5 questions)

### Q1. What are the three pillars of observability?

a) CPU, Memory, Disk
b) Metrics, Logs, Traces
c) Code, Data, Models
d) Dev, Test, Prod

**Answer:** B

---

### Q2. What is the difference between monitoring and observability?

a) No difference
b) Monitoring tracks known issues, observability explores unknown
c) Observability is simpler
d) Monitoring is deprecated

**Answer:** B

---

### Q3. What are metrics?

a) Size measurements
b) Numeric measurements over time
c) Log files
d) Code quality scores

**Answer:** B

---

### Q4. What are logs?

a) Wood records
b) Text records of discrete events
c) Metrics
d) Traces

**Answer:** B

---

### Q5. What are distributed traces?

a) Tracking code
b) Request path through distributed system
c) Log aggregation
d) Metric collection

**Answer:** B

---

## Section 2: Prometheus and Metrics (6 questions)

### Q6. What is Prometheus?

a) Greek mythology
b) Open-source monitoring and alerting system
c) Cloud provider
d) Database

**Answer:** B

---

### Q7. What query language does Prometheus use?

a) SQL
b) PromQL
c) Python
d) JavaScript

**Answer:** B

---

### Q8. What is a time series in Prometheus?

a) TV series
b) Stream of timestamped values for a metric
c) Database table
d) Log file

**Answer:** B

---

### Q9. What is the purpose of labels in Prometheus metrics?

a) Pretty printing
b) Add dimensions to metrics for filtering/aggregation
c) Color coding
d) No purpose

**Answer:** B

---

### Q10. How does Prometheus collect metrics?

a) Push model only
b) Pull model (scraping) primarily
c) Email
d) Manual upload

**Answer:** B

---

### Q11. What are the four metric types in Prometheus?

a) Small, Medium, Large, XL
b) Counter, Gauge, Histogram, Summary
c) Logs, Traces, Metrics, Events
d) CPU, Memory, Disk, Network

**Answer:** B

---

## Section 3: Logging (5 questions)

### Q12. What is structured logging?

a) Organizing log files
b) Logs in parseable format (JSON, key-value)
c) Pretty formatting
d) Log compression

**Answer:** B

---

### Q13. Which tool is commonly used for log aggregation?

a) Excel
b) ELK Stack (Elasticsearch, Logstash, Kibana)
c) Notepad
d) PowerPoint

**Answer:** B

---

### Q14. What are log levels used for?

a) Building levels
b) Categorize log severity (DEBUG, INFO, WARN, ERROR)
c) Log compression levels
d) User levels

**Answer:** B

---

### Q15. Why is centralized logging important?

a) Not important
b) Aggregate logs from distributed systems in one place
c) Save disk space
d) Make logs colorful

**Answer:** B

---

### Q16. What should you NOT log in production?

a) Errors
b) Passwords, API keys, PII
c) Request IDs
d) Timestamps

**Answer:** B

---

## Section 4: Alerting (4 questions)

### Q17. What is an alerting rule?

a) Company policy
b) Condition that triggers notification
c) Log format
d) Metric name

**Answer:** B

---

### Q18. What is alert fatigue?

a) Tired employees
b) Too many alerts causing desensitization
c) Slow alerts
d) Alert failures

**Answer:** B

---

### Q19. What is a good practice for alerts?

a) Alert on everything
b) Alert on actionable issues, not symptoms
c) Never alert
d) Alert only once per year

**Answer:** B

---

### Q20. What is the purpose of alert severity levels?

a) No purpose
b) Prioritize responses based on impact
c) Color coding
d) Log organization

**Answer:** B

---

## Section 5: ML-Specific Monitoring (5 questions)

### Q21. What model metrics should you monitor in production?

a) Nothing
b) Accuracy, latency, throughput, error rate
c) Only accuracy
d) Only latency

**Answer:** B

---

### Q22. How can you detect model drift?

a) Manual inspection daily
b) Monitor prediction distributions and performance metrics
c) Not possible
d) Only during retraining

**Answer:** B

---

### Q23. What is prediction latency?

a) Late predictions
b) Time to generate prediction
c) Model accuracy
d) Training time

**Answer:** B

---

### Q24. What is the purpose of monitoring feature distributions?

a) No purpose
b) Detect data drift and distribution shifts
c) Count features
d) Feature selection

**Answer:** B

---

### Q25. What is an SLO (Service Level Objective)?

a) Sales target
b) Target reliability metric (e.g., 99.9% uptime)
c) Team goal
d) Training objective

**Answer:** B

---

## Answer Key

1. B   2. B   3. B   4. B   5. B
6. B   7. B   8. B   9. B   10. B
11. B  12. B  13. B  14. B  15. B
16. B  17. B  18. B  19. B  20. B
21. B  22. B  23. B  24. B  25. B

---

## Scoring

- **23-25 correct (92-100%)**: Excellent! Observability expert
- **20-22 correct (80-88%)**: Good! Production-ready
- **18-19 correct (72-76%)**: Fair. Review concepts
- **Below 18 (< 72%)**: Review module materials

---

**Next Module:** Module 09 - Infrastructure as Code
