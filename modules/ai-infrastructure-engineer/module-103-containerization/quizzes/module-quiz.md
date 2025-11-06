# Module 03: Containerization with Docker - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (20/25 questions)
**Coverage:** All Lessons

---

## Section 1: Docker Fundamentals (5 questions)

### Q1. What is the main difference between a container and a virtual machine?

a) Containers are slower
b) Containers share the host OS kernel, VMs have separate OS
c) VMs use less resources
d) No difference

**Answer:** B

---

### Q2. What is a Docker image?

a) A running instance of a container
b) A read-only template used to create containers
c) A virtual machine snapshot
d) A type of volume

**Answer:** B

---

### Q3. Which command creates and starts a container from an image?

a) `docker create`
b) `docker start`
c) `docker run`
d) `docker exec`

**Answer:** C

---

### Q4. What does the `-d` flag do in `docker run -d`?

a) Delete container after exit
b) Run container in detached (background) mode
c) Debug mode
d) Development mode

**Answer:** B

---

### Q5. How do you access GPU from within a Docker container?

a) Not possible
b) Install NVIDIA Container Toolkit and use `--gpus all`
c) GPUs work automatically
d) Use `-gpu` flag

**Answer:** B

---

## Section 2: Dockerfiles for ML (6 questions)

### Q6. Which instruction in a Dockerfile specifies the base image?

a) BASE
b) FROM
c) IMAGE
d) IMPORT

**Answer:** B

---

### Q7. What is the difference between `COPY` and `ADD` in Dockerfiles?

a) No difference
b) ADD has extra features (URLs, tar extraction)
c) COPY is deprecated
d) ADD is faster

**Answer:** B

---

### Q8. What is the purpose of `WORKDIR` in a Dockerfile?

a) Set working directory for subsequent instructions
b) Create a new directory
c) Delete a directory
d) List directory contents

**Answer:** A

---

### Q9. Which base image is best for PyTorch applications with GPU support?

a) python:3.11
b) ubuntu:22.04
c) pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime
d) alpine:latest

**Answer:** C

---

### Q10. What does `CMD` specify in a Dockerfile?

a) Build-time commands
b) Default command to run when container starts
c) Copy commands
d) Compilation directives

**Answer:** B

---

### Q11. Why should you avoid using `latest` tag in production Dockerfiles?

a) It's slower
b) Non-deterministic; version can change unexpectedly
c) It's deprecated
d) No reason to avoid

**Answer:** B

---

## Section 3: Image Optimization (5 questions)

### Q12. What is a multi-stage build?

a) Building on multiple machines
b) Using multiple FROM statements to separate build and runtime
c) Building multiple images at once
d) Sequential builds

**Answer:** B

---

### Q13. What is the purpose of `.dockerignore`?

a) Ignore Docker warnings
b) Exclude files from build context
c) Hide containers
d) Prevent container startup

**Answer:** B

---

### Q14. Which technique can reduce ML Docker image size by 50-80%?

a) Compressing the image
b) Multi-stage builds and removing build dependencies
c) Using more layers
d) Adding more base images

**Answer:** B

---

### Q15. Why is layer caching important in Docker builds?

a) Increases security
b) Speeds up builds by reusing unchanged layers
c) Reduces image size
d) No real benefit

**Answer:** B

---

### Q16. Which instruction order is best for leveraging cache?

a) COPY code first, then install dependencies
b) Install dependencies first, then COPY code
c) Order doesn't matter
d) Random order is best

**Answer:** B

---

## Section 4: Networking and Volumes (5 questions)

### Q17. How do you expose a container port to the host?

a) `docker run -p 8080:80`
b) `docker run --expose 80`
c) Ports are automatically exposed
d) `docker run --network host`

**Answer:** A

---

### Q18. What is the purpose of Docker volumes?

a) Increase container speed
b) Persist data beyond container lifecycle
c) Network configuration
d) Image compression

**Answer:** B

---

### Q19. Which network mode gives containers full access to host networking?

a) bridge
b) none
c) host
d) container

**Answer:** C

---

### Q20. How can containers communicate with each other?

a) They cannot
b) Using container names on same Docker network
c) Only through the host
d) Via email

**Answer:** B

---

### Q21. What is the difference between a bind mount and a volume?

a) No difference
b) Volumes managed by Docker, bind mounts use host path
c) Bind mounts are faster
d) Volumes are deprecated

**Answer:** B

---

## Section 5: Container Registries and Best Practices (4 questions)

### Q22. What is Docker Hub?

a) A container orchestrator
b) A public container registry
c) A monitoring tool
d) A cloud provider

**Answer:** B

---

### Q23. Which command pushes an image to a registry?

a) `docker upload`
b) `docker push`
c) `docker send`
d) `docker deploy`

**Answer:** B

---

### Q24. What is a best practice for running containers in production?

a) Always run as root user
b) Run as non-root user for security
c) Use latest tag always
d) Never use volumes

**Answer:** B

---

### Q25. What does Docker Compose help with?

a) Image compression
b) Defining and running multi-container applications
c) Building images faster
d) Monitoring containers

**Answer:** B

---

## Answer Key

1. B   2. B   3. C   4. B   5. B
6. B   7. B   8. A   9. C   10. B
11. B  12. B  13. B  14. B  15. B
16. B  17. A  18. B  19. C  20. B
21. B  22. B  23. B  24. B  25. B

---

## Scoring

- **23-25 correct (92-100%)**: Excellent! Docker expert
- **20-22 correct (80-88%)**: Good! Ready for Kubernetes
- **18-19 correct (72-76%)**: Fair. Review weak areas
- **Below 18 (< 72%)**: Review module materials

---

## Areas for Review

- **Section 1**: Lesson 01 - Docker Introduction
- **Section 2**: Lesson 02 - Dockerfiles for ML Applications
- **Section 3**: Lesson 03 - Image Optimization
- **Section 4**: Lesson 04 - Networking and Volumes
- **Section 5**: Lesson 05-06 - Registries and Best Practices

---

**Next Module:** Module 04 - Kubernetes Deep Dive
