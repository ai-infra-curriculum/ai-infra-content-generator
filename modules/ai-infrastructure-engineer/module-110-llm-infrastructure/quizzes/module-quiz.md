# Module 10: LLM Infrastructure - Quiz

**Time Limit:** 35 minutes
**Passing Score:** 80% (24/30 questions)
**Coverage:** LLM deployment, optimization, serving

---

## Section 1: LLM Fundamentals (5 questions)

### Q1. What does LLM stand for?

a) Large Language Model
b) Long Learning Machine
c) Linear Logic Model
d) Low Latency Model

**Answer:** A

---

### Q2. What is the primary challenge of deploying LLMs?

a) Small model size
b) Large model size and memory requirements
c) Fast inference
d) No challenges

**Answer:** B

---

### Q3. What is the typical size of modern LLMs?

a) A few MB
b) Gigabytes to hundreds of gigabytes
c) A few KB
d) Always 1GB

**Answer:** B

---

### Q4. What is inference in LLM context?

a) Training the model
b) Generating predictions/responses
c) Data collection
d) Model compression

**Answer:** B

---

### Q5. What is a prompt in LLM usage?

a) Command prompt
b) Input text to generate response
c) Training data
d) Model parameter

**Answer:** B

---

## Section 2: LLM Serving Infrastructure (7 questions)

### Q6. Which framework is commonly used for LLM serving?

a) Excel
b) vLLM, TGI (Text Generation Inference), TensorRT-LLM
c) PowerPoint
d) Word

**Answer:** B

---

### Q7. What is the purpose of batching in LLM inference?

a) Slow down inference
b) Process multiple requests together for efficiency
c) Reduce accuracy
d) No purpose

**Answer:** B

---

### Q8. What is KV cache in LLM inference?

a) Key-Value database
b) Cache for attention keys/values to speed up generation
c) CPU cache
d) Network cache

**Answer:** B

---

### Q9. Why is GPU memory important for LLM serving?

a) Not important
b) Stores model weights and KV cache
c) Only for training
d) Storage backup

**Answer:** B

---

### Q10. What is continuous batching?

a) Batch processing at intervals
b) Dynamic batching that adds/removes requests during generation
c) Single request processing
d) No batching

**Answer:** B

---

### Q11. What is the purpose of model quantization?

a) Increase model size
b) Reduce model size and memory usage
c) Improve accuracy
d) Slow down inference

**Answer:** B

---

### Q12. What is PagedAttention in vLLM?

a) Web page attention
b) Memory-efficient attention mechanism
c) User attention tracking
d) Page ranking

**Answer:** B

---

## Section 3: LLM Optimization (6 questions)

### Q13. What is quantization?

a) Counting tokens
b) Reducing numeric precision (FP32 → INT8)
c) Increasing precision
d) Model training

**Answer:** B

---

### Q14. What are typical quantization formats?

a) PDF, DOC
b) INT8, INT4, FP16, GPTQ, AWQ
c) JPG, PNG
d) MP3, MP4

**Answer:** B

---

### Q15. What is the trade-off of quantization?

a) No trade-offs
b) Smaller size/faster inference vs potential accuracy loss
c) Always better
d) Always worse

**Answer:** B

---

### Q16. What is Flash Attention?

a) Quick thinking
b) Optimized attention algorithm for speed and memory
c) Lighting system
d) Training method

**Answer:** B

---

### Q17. What is model sharding?

a) Breaking models
b) Splitting model across multiple GPUs
c) Model backup
d) Data splitting

**Answer:** B

---

### Q18. When should you use model sharding?

a) Always
b) When model too large for single GPU
c) Never
d) Only for training

**Answer:** B

---

## Section 4: Prompt Engineering and RAG (6 questions)

### Q19. What is prompt engineering?

a) Engineering team prompts
b) Crafting effective prompts for better LLM outputs
c) System engineering
d) Network engineering

**Answer:** B

---

### Q20. What is RAG?

a) Random Access Generator
b) Retrieval Augmented Generation
c) Rapid Application Growth
d) Resource Allocation Graph

**Answer:** B

---

### Q21. What is the purpose of RAG?

a) Make models larger
b) Augment LLM with external knowledge
c) Speed up training
d) Reduce costs only

**Answer:** B

---

### Q22. What components does a RAG system need?

a) Just the LLM
b) Vector database, embedding model, LLM
c) Only database
d) Only embeddings

**Answer:** B

---

### Q23. What are embeddings in RAG context?

a) Embedded systems
b) Vector representations of text
c) Model parameters
d) Training data

**Answer:** B

---

### Q24. Which database is commonly used for RAG?

a) MySQL
b) Pinecone, Weaviate, Qdrant (vector databases)
c) Excel
d) Text files

**Answer:** B

---

## Section 5: LLM Deployment and Monitoring (6 questions)

### Q25. What metrics are important for LLM serving?

a) Only accuracy
b) Latency, throughput, tokens/sec, cost per token
c) Only cost
d) Only latency

**Answer:** B

---

### Q26. What is TTFT?

a) Time To First Tweet
b) Time To First Token
c) Total Training Time
d) Token Transfer Time

**Answer:** B

---

### Q27. What is token throughput?

a) Network bandwidth
b) Tokens generated per second
c) Training speed
d) Storage speed

**Answer:** B

---

### Q28. Why monitor GPU memory usage in LLM serving?

a) Not important
b) Prevent OOM errors, optimize batch size
c) Just for logging
d) No reason

**Answer:** B

---

### Q29. What is a best practice for LLM API deployment?

a) No rate limiting
b) Implement rate limiting, caching, monitoring
c) No monitoring
d) No caching

**Answer:** B

---

### Q30. What is the purpose of LLM serving frameworks like vLLM?

a) Training models
b) Optimize inference performance and throughput
c) Data collection
d) Model development

**Answer:** B

---

## Answer Key

1. A   2. B   3. B   4. B   5. B
6. B   7. B   8. B   9. B   10. B
11. B  12. B  13. B  14. B  15. B
16. B  17. B  18. B  19. B  20. B
21. B  22. B  23. B  24. B  25. B
26. B  27. B  28. B  29. B  30. B

---

## Scoring

- **27-30 correct (90-100%)**: Excellent! LLM infrastructure expert
- **24-26 correct (80-89%)**: Good! Production-ready
- **21-23 correct (70-79%)**: Fair. Review concepts
- **Below 21 (< 70%)**: Review module materials

---

## Module Complete!

Congratulations on completing the AI Infrastructure Engineer curriculum!

**Next Steps:**
- Complete capstone projects
- Build portfolio projects
- Proceed to Senior Engineer track
- Or specialize in MLOps, Performance, or Security tracks

---

**You've mastered:**
- Cloud infrastructure
- Containerization and Kubernetes
- Data pipelines and MLOps
- GPU computing
- Monitoring and IaC
- LLM infrastructure
