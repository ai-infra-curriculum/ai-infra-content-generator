# Migration Plan: Existing Content → New Framework

**Version**: 1.0
**Date**: 2025-11-04
**Purpose**: Migrate existing AI Infrastructure curriculum content to use the new content generator framework

---

## Executive Summary

This document provides a detailed plan for migrating existing content from the current 24-repository structure to align with the new content generator framework templates, workflows, and quality standards.

**Current State**:
- 12 learning repositories with varying content (0-20 modules each)
- 12 solutions repositories with partial implementations
- Total: 122 modules, 69 projects across all repositories
- Mixed quality and structure

**Target State**:
- All content aligned with framework templates
- Consistent structure across all 24 repositories
- Quality standards met (12,000+ words/module, 10+ examples)
- Multi-role coordination dashboard in place
- Checkpointing and validation integrated

**Migration Effort**: 80-120 hours (2-3 weeks)

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Gap Analysis](#gap-analysis)
3. [Migration Strategy](#migration-strategy)
4. [Phase-by-Phase Plan](#phase-by-phase-plan)
5. [Automated Migration Scripts](#automated-migration-scripts)
6. [Quality Validation](#quality-validation)
7. [Risk Mitigation](#risk-mitigation)
8. [Success Criteria](#success-criteria)

---

## Current State Analysis

### Content Inventory

**Learning Repositories** (as of 2025-11-04):

| Repository | Modules | Projects | Status |
|------------|---------|----------|--------|
| ai-infra-junior-engineer-learning | 10 | 5 | ✅ Substantial |
| ai-infra-engineer-learning | 10 | 3 | ✅ Substantial |
| ai-infra-senior-engineer-learning | 10 | 4 | ✅ Substantial |
| ai-infra-ml-platform-learning | 10 | 6 | ✅ Substantial |
| ai-infra-mlops-learning | 0 | 10 | ⚠️ Modules missing |
| ai-infra-security-learning | 20 | 9 | ✅ Extensive |
| ai-infra-performance-learning | 16 | 6 | ✅ Extensive |
| ai-infra-architect-learning | 10 | 5 | ✅ Substantial |
| ai-infra-senior-architect-learning | 10 | 11 | ✅ Extensive |
| ai-infra-principal-architect-learning | 10 | 4 | ✅ Substantial |
| ai-infra-principal-engineer-learning | 6 | 3 | ⚠️ Incomplete |
| ai-infra-team-lead-learning | 10 | 3 | ✅ Substantial |
| **TOTAL** | **122** | **69** | |

**Solutions Repositories**:

| Repository | Projects with Solutions | Status |
|------------|------------------------|--------|
| ai-infra-junior-engineer-solutions | 5 | ✅ Complete |
| ai-infra-engineer-solutions | 3 | ✅ Complete |
| ai-infra-senior-engineer-solutions | 4 | ✅ Complete |
| ai-infra-senior-architect-solutions | 6 | ✅ Complete |
| ai-infra-ml-platform-solutions | 0 | ❌ Empty |
| ai-infra-mlops-solutions | 0 | ❌ Empty |
| ai-infra-security-solutions | 0 | ❌ Empty |
| ai-infra-performance-solutions | 0 | ❌ Empty |
| ai-infra-architect-solutions | 5 | ✅ Complete |
| ai-infra-principal-architect-solutions | 0 | ❌ Empty |
| ai-infra-principal-engineer-solutions | 0 | ❌ Empty |
| ai-infra-team-lead-solutions | 0 | ❌ Empty |
| **TOTAL** | **23/69** | **33% coverage** |

### Content Structure Analysis

**Existing Structure** (Example: Junior Engineer):
```
ai-infra-junior-engineer-learning/
├── lessons/
│   └── mod-001-python-fundamentals/
│       ├── lecture-notes/              # Multiple MD files (5 files)
│       ├── exercises/                  # Multiple exercises
│       ├── quizzes/                    # Quiz file
│       ├── resources/                  # Additional resources
│       └── README.md                   # Module overview
```

**Framework Expected Structure**:
```
ai-infra-junior-engineer-learning/
├── lessons/
│   └── 01-foundations/
│       ├── README.md                   # Module overview
│       ├── lecture-notes.md           # SINGLE comprehensive file (12K+ words)
│       ├── exercises/                  # 5-10 structured exercises
│       └── resources.md                # Reading list & references
```

**Key Differences**:
1. **Lecture notes**: Multiple files vs single comprehensive file
2. **Naming**: `mod-XXX-name` vs `XX-name` convention
3. **Quizzes**: Separate directory vs embedded in assessments
4. **Resources**: Directory vs single markdown file
5. **Quality standards**: Variable vs 12,000+ words minimum

### Content Quality Assessment

**Sample Analysis** (mod-001-python-fundamentals):
- Total lecture words: 20,110 words ✅ (exceeds 12K minimum)
- Lecture files: 5 separate files ⚠️ (should be 1)
- Code examples: Estimated 15+ ✅
- Exercises: 10 exercises ✅
- Assessments: 1 quiz ✅
- Resources: Separate directory ⚠️

**Quality Tiers**:

**Tier 1 - Framework Ready (30% of content)**:
- Meets or exceeds word count requirements
- Has comprehensive code examples
- Exercises well-structured
- Minimal reformatting needed

**Tier 2 - Needs Restructuring (50% of content)**:
- Content quality good but structure doesn't match framework
- Multiple lecture files need consolidation
- Directory structure needs adjustment
- Reformatting primary task

**Tier 3 - Needs Enhancement (20% of content)**:
- Incomplete modules (e.g., mlops-learning has 0 modules)
- Below word count minimums
- Missing components (exercises, assessments)
- Requires new content generation

---

## Gap Analysis

### Structural Gaps

1. **Multi-Role Coordination**
   - ❌ No `curriculum/roles/multi-role-alignment.md` dashboard
   - ❌ No shared assets register
   - ❌ No cross-role dependency tracking
   - ❌ Duplicate content across roles (e.g., Docker basics in 8+ repos)

2. **Research Foundation**
   - ❌ No `research/` directory structure
   - ❌ No job posting analysis documents
   - ❌ No skills matrices in standard format
   - ❌ No practitioner interview summaries

3. **Curriculum Planning Artifacts**
   - ❌ No `curriculum/` directory with master plans
   - ❌ No module roadmaps using framework templates
   - ❌ No project plans using framework templates
   - ❌ No repository strategy configuration

4. **Solutions Structure**
   - ❌ Solutions repositories missing standard structure
   - ❌ No STEP_BY_STEP.md implementation guides
   - ❌ No ARCHITECTURE.md diagrams
   - ❌ No TROUBLESHOOTING.md guides
   - ⚠️ Only 33% of projects have solutions

5. **Validation Integration**
   - ❌ No validation scripts in repositories
   - ❌ No CI/CD workflows for content validation
   - ❌ No automated word count checks
   - ❌ No code quality gates

6. **Checkpoint System**
   - ❌ No memory/ directory for state management
   - ❌ No checkpoint save/resume scripts
   - ❌ No progress tracking metadata

### Content Gaps

1. **Learning Repositories**:
   - MLOps: 0 modules (needs 10 modules)
   - Principal Engineer: 6 modules (needs 4 more for standard 10)
   - Missing: ~14 modules worth of content

2. **Solutions Repositories**:
   - 8 repositories completely empty (ML Platform, MLOps, Security, Performance, Principal Architect, Principal Engineer, Team Lead)
   - Missing solutions for ~46 projects
   - No implementation guides for most projects

3. **Quality Gaps**:
   - Some modules below 12,000 word minimum
   - Inconsistent code example quality
   - Variable exercise depth
   - Incomplete assessments

---

## Migration Strategy

### Approach: Hybrid Migration

**Phase 1**: Structural Alignment (Non-Destructive)
- Add framework structure alongside existing content
- Don't delete or move existing files initially
- Validate migration before committing

**Phase 2**: Content Consolidation
- Merge multiple lecture files into single comprehensive files
- Restructure directories to match framework
- Preserve git history where possible

**Phase 3**: Enhancement & Gap Filling
- Generate missing content using framework
- Upgrade substandard content to meet quality bars
- Complete solution implementations

**Phase 4**: Validation & Cleanup
- Run framework validation scripts
- Remove redundant files
- Final quality checks

### Principles

1. **Preserve Existing Content** - Don't delete working content
2. **Incremental Migration** - One repository at a time
3. **Validate at Each Step** - Test before proceeding
4. **Git History** - Use `git mv` to preserve history
5. **Rollback Plan** - Tag before major changes

### Prioritization

**Priority 1 - Core Track** (Weeks 1-2):
- Junior Engineer
- Engineer
- Senior Engineer

**Priority 2 - Specialized with Content** (Week 3):
- ML Platform (10 modules)
- Security (20 modules)
- Performance (16 modules)

**Priority 3 - Architecture Track** (Week 4):
- Architect
- Senior Architect
- Principal Architect

**Priority 4 - Incomplete Roles** (Week 5):
- MLOps (0 modules → generate)
- Principal Engineer (6 modules → add 4)
- Team Lead (validate quality)

**Priority 5 - Solutions** (Week 6):
- Complete 8 empty solution repos
- Add missing implementation guides

---

## Phase-by-Phase Plan

### Phase 1: Framework Integration (Week 1)

**Goal**: Add framework structure to existing repositories without disrupting content

#### Step 1.1: Create Research & Curriculum Directories

```bash
# For each of 12 roles:
cd /project/ai-infrastructure-project

# Create framework directories
mkdir -p research/junior-engineer
mkdir -p research/engineer
mkdir -p research/senior-engineer
# ... repeat for all 12 roles

mkdir -p curriculum/junior-engineer/{modules,projects}
mkdir -p curriculum/engineer/{modules,projects}
# ... repeat for all 12 roles

mkdir -p curriculum/roles
```

#### Step 1.2: Copy Framework Templates

```bash
# Copy research templates
for role in junior-engineer engineer senior-engineer ml-platform mlops security performance architect senior-architect principal-architect principal-engineer team-lead; do
    cp ai-infra-content-generator/templates/research/role-research-template.md research/$role/
    cp ai-infra-content-generator/templates/research/job-posting-analysis-template.md research/$role/
    cp ai-infra-content-generator/templates/research/skills-matrix-template.yaml research/$role/
done

# Copy curriculum templates
for role in junior-engineer engineer senior-engineer ml-platform mlops security performance architect senior-architect principal-architect principal-engineer team-lead; do
    cp ai-infra-content-generator/templates/curriculum/master-plan-template.yaml curriculum/$role/master-plan.yaml
done

# Copy multi-role dashboard
cp ai-infra-content-generator/templates/curriculum/multi-role-alignment-template.md curriculum/roles/multi-role-alignment.md

# Copy repository strategy
cp ai-infra-content-generator/templates/curriculum/repository-strategy-template.yaml curriculum/repository-strategy.yaml
```

#### Step 1.3: Add Validation Scripts

```bash
# Copy validation tools to each learning repository
for repo in repositories/learning/ai-infra-*-learning; do
    mkdir -p $repo/.validation
    cp ai-infra-content-generator/validation/content-checkers/module-quality-checklist.md $repo/.validation/
    cp ai-infra-content-generator/validation/code-validators/validate-code-examples.py $repo/.validation/
    cp ai-infra-content-generator/validation/completeness/check-module-completeness.py $repo/.validation/
done
```

#### Step 1.4: Add CI/CD Workflows

```bash
# Add GitHub Actions for validation
for repo in repositories/learning/ai-infra-*-learning; do
    cp ai-infra-content-generator/.github/workflows/validate-content.yml $repo/.github/workflows/
done

for repo in repositories/solutions/ai-infra-*-solutions; do
    cp ai-infra-content-generator/.github/workflows/validate-solutions.yml $repo/.github/workflows/
done
```

**Deliverables**:
- ✅ Framework directory structure added
- ✅ Templates copied to appropriate locations
- ✅ Validation scripts integrated
- ✅ CI/CD workflows added
- ✅ Git committed with clear message

---

### Phase 2: Content Restructuring (Week 2-3)

**Goal**: Align existing content with framework structure

#### Step 2.1: Pilot Migration (Junior Engineer)

**Process**:
```bash
cd repositories/learning/ai-infra-junior-engineer-learning

# 1. Create new framework-compliant structure
mkdir -p lessons-framework/01-python-fundamentals

# 2. Consolidate lecture notes
cat lessons/mod-001-python-fundamentals/lecture-notes/*.md > lessons-framework/01-python-fundamentals/lecture-notes.md

# 3. Move exercises (already structured correctly)
cp -r lessons/mod-001-python-fundamentals/exercises lessons-framework/01-python-fundamentals/

# 4. Consolidate resources
cat lessons/mod-001-python-fundamentals/resources/*.md > lessons-framework/01-python-fundamentals/resources.md

# 5. Move quiz to assessments (rename)
mkdir -p lessons-framework/01-python-fundamentals/assessments
cp lessons/mod-001-python-fundamentals/quizzes/* lessons-framework/01-python-fundamentals/assessments/

# 6. Create module README
cp lessons/mod-001-python-fundamentals/README.md lessons-framework/01-python-fundamentals/README.md

# 7. Validate
python .validation/check-module-completeness.py lessons-framework/01-python-fundamentals/
python .validation/validate-code-examples.py lessons-framework/01-python-fundamentals/lecture-notes.md

# 8. If validation passes, replace old structure
git mv lessons lessons-old
git mv lessons-framework lessons
git commit -m "Migrate mod-001 to framework structure"

# 9. Test build, then delete old
rm -rf lessons-old
git commit -m "Remove old module structure after validation"
```

**Automated Script** (See Section: Automated Migration Scripts)

#### Step 2.2: Scale to All Modules

```bash
# Run migration script for all modules in Junior Engineer
./scripts/migrate-repository.sh ai-infra-junior-engineer-learning

# Validate entire repository
./scripts/validate-repository.sh ai-infra-junior-engineer-learning

# If successful, repeat for other repos
for role in engineer senior-engineer ml-platform security performance architect senior-architect principal-architect team-lead; do
    ./scripts/migrate-repository.sh ai-infra-$role-learning
done
```

#### Step 2.3: Solutions Repositories

```bash
# For repositories with existing solutions
for repo in junior-engineer-solutions engineer-solutions senior-engineer-solutions senior-architect-solutions architect-solutions; do
    ./scripts/migrate-solutions-repository.sh ai-infra-$repo
done
```

**Deliverables**:
- ✅ Junior Engineer fully migrated and validated (pilot)
- ✅ All learning repos migrated to framework structure
- ✅ Existing solutions repos restructured
- ✅ Validation passing on all migrated content

---

### Phase 3: Content Enhancement (Week 4-5)

**Goal**: Fill gaps and upgrade content to meet quality standards

#### Step 3.1: Generate Missing Modules

**MLOps Learning** (0 modules → 10 modules):
```bash
# Use framework to generate complete curriculum
cd ai-infra-content-generator

# Follow workflows/module-generation.md for each of 10 modules
# Use templates and prompts to generate:
# - Lecture notes (12,000+ words each)
# - Exercises (5-10 per module)
# - Assessments
# - Resources

# Estimate: 60 hours (6 hours per module)
```

**Principal Engineer** (6 modules → 10 modules):
```bash
# Generate 4 additional modules
# Focus on: technical leadership, strategic planning, cross-team coordination, innovation

# Estimate: 24 hours
```

#### Step 3.2: Enhance Substandard Content

**Identify modules below standards**:
```bash
# Run word count check across all repos
./scripts/check-word-counts.sh

# Modules below 12,000 words:
# - Enhance with additional sections
# - Add more code examples
# - Expand case studies
```

#### Step 3.3: Complete Solutions

**Generate solutions for 46 missing projects**:
```bash
# Priority order:
# 1. ML Platform (6 projects)
# 2. Security (9 projects)
# 3. Performance (6 projects)
# 4. MLOps (10 projects)
# 5. Others (15 projects)

# Use templates/solutions/* and prompts/solutions/*
# Estimate: 90-120 hours (2-3 hours per project solution)
```

**Deliverables**:
- ✅ MLOps learning repo complete (10 new modules)
- ✅ Principal Engineer complete (4 new modules)
- ✅ All modules meet 12K+ word minimum
- ✅ All 69 projects have solutions
- ✅ All solutions have implementation guides

---

### Phase 4: Multi-Role Coordination (Week 5)

**Goal**: Implement cross-role coordination and shared assets

#### Step 4.1: Create Multi-Role Dashboard

```bash
# Use template: templates/curriculum/multi-role-alignment-template.md
cp templates/curriculum/multi-role-alignment-template.md curriculum/roles/multi-role-alignment.md

# Populate with actual module assignments
./scripts/generate-multi-role-dashboard.py
```

**Dashboard Sections to Complete**:
1. Progression Ladder (12 roles)
2. Role Comparison Matrix
3. Module Assignment Matrix (122 modules × 12 roles)
4. Shared Assets Register
5. Cross-Role Dependencies
6. Quality Standards Compliance

#### Step 4.2: Identify Shared Assets

```bash
# Scan all repositories for duplicate content
./scripts/find-duplicate-modules.py

# Example duplicates found:
# - Docker basics: 8 repositories
# - Kubernetes intro: 10 repositories
# - Git fundamentals: 7 repositories
# - Python basics: 6 repositories
# - Linux essentials: 8 repositories

# Extract to shared/ directory
mkdir -p shared/modules/
git mv appropriate content to shared/
# Update repos to reference shared content
```

#### Step 4.3: Document Reuse Strategy

```yaml
# curriculum/repository-strategy.yaml
shared_assets:
  - name: docker-basics
    location: shared/modules/docker-basics/
    used_by:
      - junior-engineer
      - engineer
      - senior-engineer
      - ml-platform
      - mlops
      - security
      - performance
      - architect
    maintenance_owner: junior-engineer

  - name: kubernetes-intro
    location: shared/modules/kubernetes-intro/
    used_by:
      - engineer
      - senior-engineer
      - ml-platform
      - mlops
      - security
      - performance
      - architect
      - senior-architect
      - principal-architect
      - principal-engineer
    maintenance_owner: engineer
```

**Deliverables**:
- ✅ Multi-role dashboard complete and accurate
- ✅ Shared assets identified and extracted
- ✅ Repository strategy documented
- ✅ Cross-role dependencies mapped

---

### Phase 5: Validation & Quality Assurance (Week 6)

**Goal**: Comprehensive validation of all migrated content

#### Step 5.1: Automated Validation

```bash
# Run validation suite on all repositories
for repo in repositories/learning/ai-infra-*-learning; do
    echo "Validating $repo..."
    ./scripts/validate-repository.sh $repo
done

# Check all solutions
for repo in repositories/solutions/ai-infra-*-solutions; do
    echo "Validating $repo..."
    ./scripts/validate-solutions.sh $repo
done

# Generate validation report
./scripts/generate-validation-report.py > validation-report.md
```

#### Step 5.2: Manual Quality Review

**Checklist per Repository**:
- [ ] All modules have 12,000+ words
- [ ] All modules have 10+ code examples
- [ ] All modules have 3+ case studies
- [ ] All modules have 5-10 exercises
- [ ] All modules have assessments
- [ ] All projects have complete solutions
- [ ] All solutions have implementation guides
- [ ] All code passes linting
- [ ] All tests pass
- [ ] Documentation complete

#### Step 5.3: Fix Validation Issues

```bash
# Issues found during validation:
# - 5 modules below word count → enhance
# - 12 modules missing code examples → add examples
# - 8 projects missing tests → write tests
# - 15 solutions missing guides → write guides

# Estimate: 40 hours for fixes
```

**Deliverables**:
- ✅ Validation report showing 100% compliance
- ✅ All quality gates passing
- ✅ All CI/CD workflows green
- ✅ Documentation complete and accurate

---

### Phase 6: Migration Completion (Week 6)

**Goal**: Final cleanup and documentation

#### Step 6.1: Remove Old Artifacts

```bash
# Clean up old structure remnants
for repo in repositories/learning/ai-infra-*-learning; do
    cd $repo
    # Remove any *-old directories
    rm -rf lessons-old resources-old
    # Remove migration artifacts
    rm -rf .migration/
    git commit -m "Clean up migration artifacts"
done
```

#### Step 6.2: Update Documentation

```bash
# Update README files to reference framework
for repo in repositories/learning/ai-infra-*-learning; do
    cd $repo
    # Add framework compliance badge
    # Update structure documentation
    # Add links to framework templates
    git commit -m "Update documentation for framework compliance"
done
```

#### Step 6.3: Create Migration Report

```markdown
# MIGRATION_REPORT.md

## Summary
- Start Date: 2025-11-04
- Completion Date: 2025-11-XX
- Repositories Migrated: 24
- Modules Migrated: 122
- Modules Generated: 14 new
- Projects with Solutions: 69 (100%)
- Total Effort: XXX hours

## Changes
- [List all major changes]

## Validation Results
- [Attach validation reports]

## Known Issues
- [Document any remaining issues]

## Next Steps
- [Recommendations for ongoing maintenance]
```

**Deliverables**:
- ✅ All old artifacts removed
- ✅ Documentation updated
- ✅ Migration report published
- ✅ Framework fully adopted

---

## Automated Migration Scripts

### Script 1: migrate-repository.sh

```bash
#!/bin/bash
# migrate-repository.sh - Migrate a learning repository to framework structure

REPO_PATH=$1
REPO_NAME=$(basename "$REPO_PATH")

if [ -z "$REPO_PATH" ]; then
    echo "Usage: ./migrate-repository.sh <repo-path>"
    exit 1
fi

echo "Migrating repository: $REPO_NAME"
cd "$REPO_PATH"

# Backup
git tag "pre-migration-$(date +%Y%m%d)"

# Create framework structure
mkdir -p lessons-framework

# Migrate each module
for module_dir in lessons/mod-*/; do
    if [ ! -d "$module_dir" ]; then
        continue
    fi

    module_name=$(basename "$module_dir")
    # Extract number and name (mod-001-python-fundamentals → 01-python-fundamentals)
    new_name=$(echo "$module_name" | sed 's/mod-0*\([0-9]*\)-/\1-/')

    echo "Migrating module: $module_name → $new_name"

    mkdir -p "lessons-framework/$new_name"

    # Consolidate lecture notes
    if [ -d "$module_dir/lecture-notes" ]; then
        cat "$module_dir/lecture-notes"/*.md > "lessons-framework/$new_name/lecture-notes.md"
    fi

    # Copy exercises
    if [ -d "$module_dir/exercises" ]; then
        cp -r "$module_dir/exercises" "lessons-framework/$new_name/"
    fi

    # Consolidate resources
    if [ -d "$module_dir/resources" ]; then
        cat "$module_dir/resources"/*.md > "lessons-framework/$new_name/resources.md" 2>/dev/null || touch "lessons-framework/$new_name/resources.md"
    fi

    # Move quiz to assessments
    mkdir -p "lessons-framework/$new_name/assessments"
    if [ -d "$module_dir/quizzes" ]; then
        cp "$module_dir/quizzes"/* "lessons-framework/$new_name/assessments/" 2>/dev/null
    fi
    if [ -d "$module_dir/quiz" ]; then
        cp "$module_dir/quiz"/* "lessons-framework/$new_name/assessments/" 2>/dev/null
    fi

    # Copy README
    if [ -f "$module_dir/README.md" ]; then
        cp "$module_dir/README.md" "lessons-framework/$new_name/"
    fi
done

# Validate
echo "Running validation..."
python .validation/check-module-completeness.py lessons-framework/

# If validation passes, replace
if [ $? -eq 0 ]; then
    echo "Validation passed. Replacing old structure..."
    git mv lessons lessons-old
    git mv lessons-framework lessons
    git commit -m "Migrate $REPO_NAME to framework structure"

    echo "Testing repository..."
    # Run any tests

    echo "Cleaning up old structure..."
    rm -rf lessons-old
    git commit -m "Remove old structure after validation"

    echo "Migration complete!"
else
    echo "Validation failed. Review lessons-framework/ and fix issues."
    exit 1
fi
```

### Script 2: generate-multi-role-dashboard.py

```python
#!/usr/bin/env python3
"""
generate-multi-role-dashboard.py
Scans all repositories and generates multi-role alignment dashboard
"""

import os
import glob
import yaml

ROLES = [
    "junior-engineer",
    "engineer",
    "senior-engineer",
    "ml-platform",
    "mlops",
    "security",
    "performance",
    "architect",
    "senior-architect",
    "principal-architect",
    "principal-engineer",
    "team-lead"
]

def scan_modules(role):
    """Scan modules for a given role"""
    repo_path = f"repositories/learning/ai-infra-{role}-learning/lessons"

    if not os.path.exists(repo_path):
        return []

    modules = []
    for module_dir in sorted(os.listdir(repo_path)):
        if os.path.isdir(os.path.join(repo_path, module_dir)):
            modules.append(module_dir)

    return modules

def generate_dashboard():
    """Generate multi-role alignment dashboard"""

    # Collect all modules per role
    role_modules = {}
    all_modules = set()

    for role in ROLES:
        modules = scan_modules(role)
        role_modules[role] = modules
        all_modules.update(modules)

    # Generate markdown
    output = """# Multi-Role Alignment Dashboard

Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

---

## Module Assignment Matrix

| Module | """ + " | ".join([r.replace("-", " ").title() for r in ROLES]) + """ |
|--------|""" + "|".join(["--------"] * len(ROLES)) + """|
"""

    for module in sorted(all_modules):
        row = f"| {module} |"
        for role in ROLES:
            if module in role_modules[role]:
                row += " ✅ |"
            else:
                row += " ❌ |"
        output += row + "\n"

    output += """
---

## Shared Assets Register

| Asset | Used By (Count) | Roles | Maintenance Owner |
|-------|-----------------|-------|-------------------|
"""

    # Find shared modules (in 3+ roles)
    shared_assets = []
    for module in all_modules:
        roles_with_module = [r for r in ROLES if module in role_modules[r]]
        if len(roles_with_module) >= 3:
            shared_assets.append((module, len(roles_with_module), roles_with_module))

    for asset, count, roles in sorted(shared_assets, key=lambda x: x[1], reverse=True):
        roles_str = ", ".join(roles[:3]) + ("..." if len(roles) > 3 else "")
        owner = roles[0]  # First role is owner
        output += f"| {asset} | {count} | {roles_str} | {owner} |\n"

    output += """
---

## Statistics

"""
    total_modules = sum(len(modules) for modules in role_modules.values())
    output += f"- Total Modules: {total_modules}\n"
    output += f"- Unique Modules: {len(all_modules)}\n"
    output += f"- Shared Modules (3+ roles): {len(shared_assets)}\n"
    output += f"- Duplication Rate: {((total_modules - len(all_modules)) / total_modules * 100):.1f}%\n"

    return output

if __name__ == "__main__":
    from datetime import datetime

    dashboard = generate_dashboard()

    with open("curriculum/roles/multi-role-alignment.md", "w") as f:
        f.write(dashboard)

    print("Dashboard generated: curriculum/roles/multi-role-alignment.md")
```

### Script 3: validate-repository.sh

```bash
#!/bin/bash
# validate-repository.sh - Comprehensive validation for a repository

REPO_PATH=$1

if [ -z "$REPO_PATH" ]; then
    echo "Usage: ./validate-repository.sh <repo-path>"
    exit 1
fi

echo "Validating repository: $REPO_PATH"
cd "$REPO_PATH"

ERRORS=0

# 1. Structure validation
echo "Checking directory structure..."
[ -d "lessons" ] || { echo "❌ Missing lessons/"; ERRORS=$((ERRORS + 1)); }
[ -d "projects" ] || { echo "❌ Missing projects/"; ERRORS=$((ERRORS + 1)); }
[ -d ".github" ] || { echo "❌ Missing .github/"; ERRORS=$((ERRORS + 1)); }
[ -f "README.md" ] || { echo "❌ Missing README.md"; ERRORS=$((ERRORS + 1)); }

# 2. Module validation
echo "Validating modules..."
for module in lessons/*/; do
    if [ ! -d "$module" ]; then
        continue
    fi

    module_name=$(basename "$module")
    echo "  Checking $module_name..."

    # Check required files
    [ -f "$module/lecture-notes.md" ] || { echo "    ❌ Missing lecture-notes.md"; ERRORS=$((ERRORS + 1)); }
    [ -d "$module/exercises" ] || { echo "    ❌ Missing exercises/"; ERRORS=$((ERRORS + 1)); }

    # Check word count
    if [ -f "$module/lecture-notes.md" ]; then
        word_count=$(wc -w < "$module/lecture-notes.md")
        if [ $word_count -lt 12000 ]; then
            echo "    ❌ Word count too low: $word_count (min 12000)"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✅ Word count: $word_count"
        fi
    fi

    # Check exercises
    if [ -d "$module/exercises" ]; then
        exercise_count=$(ls -1 "$module/exercises" | wc -l)
        if [ $exercise_count -lt 5 ]; then
            echo "    ⚠️  Low exercise count: $exercise_count (recommended 5-10)"
        else
            echo "    ✅ Exercises: $exercise_count"
        fi
    fi
done

# 3. Code validation
echo "Running code quality checks..."
if command -v pylint &> /dev/null; then
    find . -name "*.py" -type f -exec pylint {} \; || ERRORS=$((ERRORS + 1))
fi

# 4. Test execution
echo "Running tests..."
if [ -d "tests" ]; then
    pytest tests/ || ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "==================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ Validation PASSED"
    exit 0
else
    echo "❌ Validation FAILED with $ERRORS errors"
    exit 1
fi
```

---

## Quality Validation

### Pre-Migration Validation

**Checklist Before Starting Migration**:
- [ ] Backup all repositories (git tags + external backup)
- [ ] MCP servers installed and tested
- [ ] Migration scripts tested on sample repository
- [ ] Validation scripts working
- [ ] Rollback procedure documented
- [ ] Team notified of migration plan

### During Migration Validation

**After Each Repository**:
- [ ] Structure matches framework
- [ ] All modules validated
- [ ] Word counts meet minimums
- [ ] Code examples tested
- [ ] Exercises functional
- [ ] Git history preserved
- [ ] CI/CD workflows passing

### Post-Migration Validation

**Final Checks**:
- [ ] All 24 repositories migrated
- [ ] Multi-role dashboard accurate
- [ ] Shared assets identified
- [ ] Solutions complete
- [ ] Documentation updated
- [ ] Quality standards met
- [ ] Migration report published

---

## Risk Mitigation

### Risk 1: Content Loss During Migration

**Mitigation**:
- Git tag before migration: `git tag pre-migration-20251104`
- External backup: `rsync -av repositories/ /backup/`
- Validation before deletion of old structure
- Keep old structure for 30 days after successful migration

### Risk 2: Migration Script Bugs

**Mitigation**:
- Test on single module first
- Test on single repository second
- Manual review of migrated content
- Parallel validation (old vs new)
- Rollback capability

### Risk 3: Breaking Existing Links

**Mitigation**:
- Document all URL changes
- Set up redirects if published
- Update cross-references
- Test all internal links

### Risk 4: Validation False Positives

**Mitigation**:
- Manual spot-checks
- Multiple validation methods
- Human review of edge cases
- Feedback loop for script improvements

### Risk 5: Incomplete Solutions

**Mitigation**:
- Track solution completion separately
- Prioritize high-value solutions
- Quality over quantity
- Use framework templates consistently

---

## Success Criteria

### Phase-Level Success Criteria

**Phase 1: Framework Integration**
- ✅ All templates copied to appropriate locations
- ✅ Validation scripts integrated
- ✅ CI/CD workflows added
- ✅ No disruption to existing content

**Phase 2: Content Restructuring**
- ✅ Junior Engineer pilot successful
- ✅ All repositories migrated
- ✅ Structure matches framework
- ✅ Validation passing

**Phase 3: Content Enhancement**
- ✅ MLOps complete (10 new modules)
- ✅ Principal Engineer complete
- ✅ All modules ≥12K words
- ✅ All solutions implemented

**Phase 4: Multi-Role Coordination**
- ✅ Dashboard complete
- ✅ Shared assets extracted
- ✅ Reuse strategy documented
- ✅ Dependencies mapped

**Phase 5: Validation & QA**
- ✅ 100% validation passing
- ✅ All quality gates green
- ✅ Manual review complete
- ✅ Issues resolved

**Phase 6: Migration Completion**
- ✅ Old artifacts removed
- ✅ Documentation updated
- ✅ Migration report published
- ✅ Framework fully adopted

### Overall Success Metrics

**Quantitative**:
- 122 modules migrated + 14 new = 136 total
- 69 projects with 100% solution coverage
- 100% modules ≥12,000 words
- 100% CI/CD workflows passing
- 0 critical validation errors

**Qualitative**:
- Consistent structure across all repositories
- Framework compliance verified
- Quality standards met
- Multi-role coordination active
- Maintenance procedures documented

---

## Timeline Summary

| Week | Phase | Focus | Effort |
|------|-------|-------|--------|
| 1 | Phase 1 | Framework Integration | 20h |
| 2-3 | Phase 2 | Content Restructuring | 40h |
| 4-5 | Phase 3 | Content Enhancement | 80h |
| 5 | Phase 4 | Multi-Role Coordination | 20h |
| 6 | Phase 5 | Validation & QA | 30h |
| 6 | Phase 6 | Migration Completion | 10h |
| **TOTAL** | **6 weeks** | **All Phases** | **200h** |

---

## Conclusion

This migration plan provides a comprehensive, phased approach to migrating existing AI Infrastructure curriculum content to the new content generator framework. By following this plan:

1. **Existing content is preserved** - Nothing is lost
2. **Quality is improved** - All content meets framework standards
3. **Structure is consistent** - All repositories follow same patterns
4. **Gaps are filled** - Missing content generated
5. **Coordination is enabled** - Multi-role dashboard tracks all content
6. **Maintenance is simplified** - Framework tools integrated

The migration is designed to be **incremental**, **validated**, and **reversible** at each step. With proper execution, the result will be a cohesive, high-quality curriculum that fully leverages the content generator framework.

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-04
**Version**: 1.0
**Contact**: ai-infra-curriculum@joshua-ferguson.com
