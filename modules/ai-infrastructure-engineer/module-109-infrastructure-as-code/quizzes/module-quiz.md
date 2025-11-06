# Module 09: Infrastructure as Code - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (20/25 questions)
**Coverage:** Terraform, IaC principles, automation

---

## Section 1: IaC Fundamentals (5 questions)

### Q1. What is Infrastructure as Code (IaC)?

a) Writing code on infrastructure
b) Managing infrastructure through code/configuration files
c) Infrastructure documentation
d) Code hosting service

**Answer:** B

---

### Q2. What are the benefits of IaC?

a) More complexity
b) Reproducibility, version control, automation
c) Slower deployments
d) No benefits

**Answer:** B

---

### Q3. What is declarative vs imperative IaC?

a) No difference
b) Declarative defines desired state, imperative defines steps
c) Declarative is slower
d) Imperative is better

**Answer:** B

---

### Q4. Which is an example of declarative IaC tool?

a) Bash scripts
b) Terraform, CloudFormation
c) Python scripts
d) Manual configuration

**Answer:** B

---

### Q5. What is idempotency in IaC?

a) Unique operations
b) Same operation produces same result repeatedly
c) Fast execution
d) Error handling

**Answer:** B

---

## Section 2: Terraform Basics (6 questions)

### Q6. What is Terraform?

a) Earth formation tool
b) Open-source IaC tool by HashiCorp
c) Cloud provider
d) Container orchestrator

**Answer:** B

---

### Q7. What is HCL in Terraform?

a) High-Level Code
b) HashiCorp Configuration Language
c) HTML Code Language
d) Hypertext Configuration

**Answer:** B

---

### Q8. What does `terraform init` do?

a) Initialize cloud account
b) Initialize working directory, download providers
c) Create infrastructure
d) Destroy infrastructure

**Answer:** B

---

### Q9. What does `terraform plan` do?

a) Create resources
b) Show planned changes without applying
c) Destroy resources
d) Format code

**Answer:** B

---

### Q10. What does `terraform apply` do?

a) Just plan changes
b) Apply planned changes to create/update infrastructure
c) Delete everything
d) Format files

**Answer:** B

---

### Q11. What is Terraform state?

a) US state
b) File tracking managed infrastructure
c) Cloud region
d) Error state

**Answer:** B

---

## Section 3: Terraform Resources and Modules (5 questions)

### Q12. What is a resource in Terraform?

a) Cloud credits
b) Infrastructure component to manage
c) Documentation
d) Variable

**Answer:** B

---

### Q13. What is a Terraform module?

a) Python module
b) Reusable Terraform configuration
c) Cloud service
d) Container

**Answer:** B

---

### Q14. What is a Terraform provider?

a) Cloud company
b) Plugin that manages specific infrastructure type
c) Data provider
d) Network provider

**Answer:** B

---

### Q15. How do you reference resources in Terraform?

a) By ID number
b) Using resource type and name (e.g., aws_instance.example)
c) By IP address
d) Random selection

**Answer:** B

---

### Q16. What are Terraform variables used for?

a) Mathematics
b) Parameterize configurations for reusability
c) Store state
d) Error messages

**Answer:** B

---

## Section 4: State Management (4 questions)

### Q17. Why is Terraform state important?

a) Not important
b) Maps configuration to real infrastructure
c) Stores passwords
d) Backup only

**Answer:** B

---

### Q18. Where should you store Terraform state for teams?

a) Local machine
b) Remote backend (S3, GCS, Terraform Cloud)
c) Email
d) USB drive

**Answer:** B

---

### Q19. What is state locking?

a) Encrypting state
b) Preventing concurrent state modifications
c) Freezing infrastructure
d) Access control only

**Answer:** B

---

### Q20. What command refreshes state with real infrastructure?

a) `terraform update`
b) `terraform refresh`
c) `terraform sync`
d) `terraform reload`

**Answer:** B

---

## Section 5: Best Practices and Advanced (5 questions)

### Q21. What is the purpose of `.tfvars` files?

a) Store temporary files
b) Store variable values separately from code
c) Terraform binary
d) Log files

**Answer:** B

---

### Q22. Should you commit terraform.tfstate to Git?

a) Yes, always
b) No, use remote backend instead
c) Only for small projects
d) Only encrypted

**Answer:** B

---

### Q23. What is a Terraform workspace?

a) Physical office
b) Isolated instance of state for same configuration
c) Code editor
d) Cloud account

**Answer:** B

---

### Q24. What is `terraform destroy` used for?

a) Delete Terraform
b) Destroy managed infrastructure
c) Delete state file
d) Format code

**Answer:** B

---

### Q25. Best practice for organizing Terraform code?

a) Single file for everything
b) Separate by environment, use modules, version control
c) Random organization
d) Never organize

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

- **23-25 correct (92-100%)**: Excellent! IaC expert
- **20-22 correct (80-88%)**: Good! Ready for production
- **18-19 correct (72-76%)**: Fair. Review concepts
- **Below 18 (< 72%)**: Review module materials

---

**Next Module:** Module 10 - LLM Infrastructure
