# Module 01: Foundations - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (24/30 questions)

---

## Section 1: ML Infrastructure Overview (5 questions)

### Q1. What is the primary role of an AI Infrastructure Engineer?

a) Developing machine learning algorithms
b) Building and maintaining infrastructure for ML systems
c) Collecting and labeling training data
d) Creating data visualizations

**Answer:** B

---

### Q2. Which of the following is NOT a typical responsibility of an AI Infrastructure Engineer?

a) Deploying models to production
b) Setting up monitoring and observability
c) Annotating training data
d) Managing GPU clusters

**Answer:** C

---

### Q3. What is the difference between MLOps and traditional DevOps?

a) MLOps is only for Python applications
b) MLOps includes data versioning and model management
c) MLOps doesn't use containers
d) There is no difference

**Answer:** B

---

### Q4. Which technology is commonly used for container orchestration in ML infrastructure?

a) Jenkins
b) Kubernetes
c) MySQL
d) Apache Kafka

**Answer:** B

---

### Q5. What is model drift?

a) When model files become corrupted
b) When model performance degrades over time
c) When models are deployed to wrong servers
d) When training takes too long

**Answer:** B

---

## Section 2: Python for Infrastructure (10 questions)

### Q6. Which Python version is recommended for modern ML infrastructure?

a) Python 2.7
b) Python 3.6
c) Python 3.9+
d) Python 4.0

**Answer:** C

---

### Q7. What is the purpose of a virtual environment in Python?

a) To run Python code in the cloud
b) To isolate project dependencies
c) To make Python run faster
d) To encrypt Python code

**Answer:** B

---

### Q8. Which command creates a virtual environment in Python 3?

a) `python3 -m venv myenv`
b) `virtualenv create myenv`
c) `pip install venv myenv`
d) `python3 --create-env myenv`

**Answer:** A

---

### Q9. What file lists Python package dependencies?

a) packages.txt
b) dependencies.json
c) requirements.txt
d) install.cfg

**Answer:** C

---

### Q10. Which library is commonly used for async HTTP requests in Python?

a) requests
b) aiohttp
c) urllib
d) http.client

**Answer:** B

---

### Q11. What is the purpose of type hints in Python?

a) Required for code to run
b) Improve code readability and enable static analysis
c) Make Python code run faster
d) Prevent runtime errors completely

**Answer:** B

---

### Q12. Which testing framework is most popular for Python?

a) JUnit
b) pytest
c) unittest only
d) TestNG

**Answer:** B

---

### Q13. What does `pip freeze > requirements.txt` do?

a) Deletes unused packages
b) Saves current installed packages to file
c) Updates all packages
d) Freezes Python version

**Answer:** B

---

### Q14. Which Python library is used for structured logging?

a) print()
b) logging
c) structlog
d) Both b and c

**Answer:** D

---

### Q15. What is a decorator in Python?

a) A way to make code look pretty
b) A function that modifies another function
c) A type of class
d) A special variable

**Answer:** B

---

## Section 3: Docker Basics (8 questions)

### Q16. What is a Docker container?

a) A virtual machine
b) An isolated process with its own filesystem
c) A cloud server
d) A type of database

**Answer:** B

---

### Q17. What file defines how to build a Docker image?

a) docker-compose.yml
b) Dockerfile
c) build.txt
d) image.json

**Answer:** B

---

### Q18. Which command builds a Docker image?

a) `docker create -t myimage .`
b) `docker build -t myimage .`
c) `docker make myimage`
d) `docker compile myimage`

**Answer:** B

---

### Q19. What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?

a) No difference
b) CMD provides default arguments, ENTRYPOINT defines the executable
c) ENTRYPOINT is deprecated
d) CMD is for Linux, ENTRYPOINT is for Windows

**Answer:** B

---

### Q20. Which layer in a Docker image is read-write?

a) Base layer
b) Container layer (top layer)
c) All layers are read-write
d) None, images are read-only

**Answer:** B

---

### Q21. What is the purpose of .dockerignore?

a) Ignore Docker warnings
b) Exclude files from build context
c) Prevent container from starting
d) Hide containers from `docker ps`

**Answer:** B

---

### Q22. Which is a best practice for Docker images?

a) Use `latest` tag always
b) Run as root user
c) Use multi-stage builds to reduce size
d) Include development tools in production image

**Answer:** C

---

### Q23. What does `docker-compose` help with?

a) Composing Dockerfiles
b) Running multiple containers together
c) Compressing Docker images
d) Creating Docker registries

**Answer:** B

---

## Section 4: Git & Version Control (5 questions)

### Q24. What is the purpose of branching in Git?

a) To delete old code
b) To work on features independently
c) To compress repository
d) To backup code

**Answer:** B

---

### Q25. Which command creates a new branch and switches to it?

a) `git branch new-feature && git checkout new-feature`
b) `git checkout -b new-feature`
c) `git create-branch new-feature`
d) Both a and b

**Answer:** D

---

### Q26. What is a merge conflict?

a) When two branches can't be found
b) When same lines are changed in different branches
c) When Git crashes
d) When remote repository is down

**Answer:** B

---

### Q27. What does `git pull` do?

a) Fetches and merges changes from remote
b) Pushes local changes to remote
c) Deletes remote branch
d) Creates new repository

**Answer:** A

---

### Q28. What should you NEVER commit to Git?

a) Source code
b) Configuration files
c) Secrets and API keys
d) Documentation

**Answer:** C

---

## Section 5: Best Practices (2 questions)

### Q29. Which is a best practice for ML infrastructure code?

a) Hardcode all configuration
b) Use environment variables for config
c) Never log errors
d) Run everything as root

**Answer:** B

---

### Q30. What is Infrastructure as Code (IaC)?

a) Writing infrastructure in assembly language
b) Managing infrastructure through code/configuration files
c) Coding only on cloud infrastructure
d) Using AI to write infrastructure code

**Answer:** B

---

## Answer Key

1. B  2. C  3. B  4. B  5. B
6. C  7. B  8. A  9. C  10. B
11. B  12. B  13. B  14. D  15. B
16. B  17. B  18. B  19. B  20. B
21. B  22. C  23. B  24. B  25. D
26. B  27. A  28. C  29. B  30. B

---

## Scoring

- **27-30 correct (90-100%)**: Excellent! Strong foundation
- **24-26 correct (80-89%)**: Good! Ready to proceed
- **21-23 correct (70-79%)**: Fair. Review weak areas
- **Below 21 (< 70%)**: Review module materials before continuing

---

## Areas for Review

If you scored poorly in a section, review these resources:

- **Section 1**: README.md, ML Infrastructure Overview lecture
- **Section 2**: Python for Infrastructure lessons, Python docs
- **Section 3**: Docker Fundamentals lesson, Docker docs
- **Section 4**: Git lesson, Git cheat sheet
- **Section 5**: Best Practices lecture notes

---

**Note:** This quiz is for self-assessment. Take your time and use it to identify knowledge gaps!
