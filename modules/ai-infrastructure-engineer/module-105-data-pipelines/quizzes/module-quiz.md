# Module 05: Data Pipelines for ML - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (20/25 questions)
**Coverage:** Data pipelines, ETL, orchestration

---

## Section 1: Data Pipeline Fundamentals (5 questions)

### Q1. What is ETL?

a) Embedded Testing Language
b) Extract, Transform, Load
c) Efficient Transfer Layer
d) Error Tracking Log

**Answer:** B

---

### Q2. What is the difference between ETL and ELT?

a) No difference
b) ETL transforms before loading, ELT loads then transforms
c) ELT is deprecated
d) ETL is only for ML

**Answer:** B

---

### Q3. What is a data pipeline?

a) Physical pipe for data
b) Automated workflow for moving and transforming data
c) A type of database
d) A network protocol

**Answer:** B

---

### Q4. Why is data versioning important in ML pipelines?

a) It's not important
b) Enables reproducibility and debugging
c) Saves storage space
d) Makes training faster

**Answer:** B

---

### Q5. What is a common cause of ML pipeline failures?

a) Too much documentation
b) Data quality issues and schema changes
c) Using Python
d) Having backups

**Answer:** B

---

## Section 2: Apache Airflow (6 questions)

### Q6. What is Apache Airflow?

a) A cloud provider
b) Workflow orchestration platform
c) A database
d) A container runtime

**Answer:** B

---

### Q7. What is a DAG in Airflow?

a) Data Analysis Graph
b) Directed Acyclic Graph defining workflow
c) Database Access Gateway
d) Deployment Automation Guide

**Answer:** B

---

### Q8. What does "acyclic" mean in DAG?

a) Can have cycles
b) No circular dependencies/loops
c) Very fast
d) Requires cycles

**Answer:** B

---

### Q9. What is an Airflow Operator?

a) Human operator
b) Task template defining what to execute
c) Mathematical operator
d) Network operator

**Answer:** B

---

### Q10. What is the purpose of task dependencies in Airflow?

a) Make tasks slower
b) Define execution order and relationships
c) No real purpose
d) Increase complexity

**Answer:** B

---

### Q11. How do you trigger an Airflow DAG manually?

a) Restart the server
b) Use UI or CLI to trigger
c) Not possible
d) Wait for schedule

**Answer:** B

---

## Section 3: Data Quality and Validation (4 questions)

### Q12. What is data drift?

a) Data moving slowly
b) Statistical properties of data changing over time
c) Data corruption
d) Network latency

**Answer:** B

---

### Q13. What is schema validation?

a) Validating database schemas
b) Checking data conforms to expected structure
c) Validating code
d) Network validation

**Answer:** B

---

### Q14. What is Great Expectations?

a) A motivational tool
b) Python library for data validation
c) A cloud service
d) A testing framework for code

**Answer:** B

---

### Q15. Why validate data before training ML models?

a) Not necessary
b) Prevent training on bad data that degrades model quality
c) Slow down pipeline
d) Increase costs

**Answer:** B

---

## Section 4: Batch vs Streaming (5 questions)

### Q16. What is batch processing?

a) Processing data in real-time
b) Processing data in large chunks at intervals
c) Processing one record at a time
d) No processing

**Answer:** B

---

### Q17. What is stream processing?

a) Processing historical data
b) Processing data continuously as it arrives
c) Batch processing with different name
d) Video processing

**Answer:** B

---

### Q18. Which tool is commonly used for stream processing?

a) Excel
b) Apache Kafka, Apache Flink
c) Jupyter Notebook
d) Git

**Answer:** B

---

### Q19. When should you use batch processing for ML?

a) Never
b) For offline model training on historical data
c) For real-time predictions
d) Always

**Answer:** B

---

### Q20. When is stream processing preferred?

a) For historical analysis only
b) For real-time features and predictions
c) Never
d) Only for small data

**Answer:** B

---

## Section 5: Data Storage and Feature Engineering (5 questions)

### Q21. What is a data lake?

a) A relational database
b) Repository storing raw data in native format
c) A caching system
d) A backup solution

**Answer:** B

---

### Q22. What is feature engineering?

a) Building features in UI
b) Creating ML model inputs from raw data
c) Engineering team features
d) Software features

**Answer:** B

---

### Q23. What is a feature store?

a) App store for features
b) Centralized repository for ML features
c) Storage for models
d) Data warehouse

**Answer:** B

---

### Q24. Why use a feature store?

a) Not useful
b) Share features across teams, ensure consistency
c) Increase complexity
d) Slow down development

**Answer:** B

---

### Q25. What format is commonly used for efficient ML data storage?

a) CSV only
b) Parquet, TFRecord, Arrow
c) Word documents
d) JSON only

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

- **23-25 correct (92-100%)**: Excellent! Pipeline expert
- **20-22 correct (80-88%)**: Good! Ready for MLOps
- **18-19 correct (72-76%)**: Fair. Review weak areas
- **Below 18 (< 72%)**: Review module materials

---

**Next Module:** Module 06 - MLOps
