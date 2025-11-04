# AI Infrastructure Curriculum End-to-End Guide

This guide demonstrates how to generate every artifact required for the AI Infrastructure Curriculum Project using the content generator framework. It stitches together templates, prompts, and workflows so a program team (or a coordinated agent team) can produce multi-role curricula with validated learning and solution assets.

---

## Overview

| Phase | Goal | Primary Outputs | Key References |
|-------|------|-----------------|----------------|
| Phase 0 | Prepare repositories, tooling, and source material | Repo strategy, workspace scaffold | `workflows/multi-role-program.md`, `templates/curriculum/repository-strategy-template.yaml` |
| Phase 1 | Conduct role research | Role briefs, skills matrices, interviews | `templates/research/*`, `prompts/research/*`, `research/README.md` |
| Phase 2 | Design curricula per role | Master plans, module roadmaps, project plans | `templates/curriculum/*`, `workflows/curriculum-design.md`, `curriculum/README.md` |
| Phase 3 | Generate learning content | Lecture notes, exercises, assessments | `workflows/module-generation.md`, `templates/lecture-notes/*`, `templates/exercises/*`, `templates/assessments/*` |
| Phase 4 | Build project assets | Hands-on project briefs & guides | `workflows/project-generation.md`, `templates/projects/*`, `templates/curriculum/project-plan-template.md` |
| Phase 5 | Produce solutions repositories | Exercise / project / assessment solutions | `templates/solutions/*`, `prompts/solutions/solution-generation-prompt.md` |
| Phase 6 | Validate and publish | QA reports, release notes | `validation/*`, `CHANGELOG.md`, release processes |

---

## Phase 0 – Program Setup

1. **Clone the framework**
   ```bash
   git clone https://github.com/ai-infra-curriculum/ai-infra-content-generator.git
   cd ai-infra-content-generator
   ```

2. **Define repository strategy**
   - Copy `templates/curriculum/repository-strategy-template.yaml` to `curriculum/repository-strategy.yaml`.
   - Decide:
     - `repositories.mode`: `single_repo` if all roles share a mono-repo; `per_role` for separate repos.
     - `solutions.placement`: `inline` vs `separate`.
   - Fill in shared components (`shared_assets`) and progression rules to encourage reuse.

3. **Mirroring repositories**
   - Create (or reserve) GitHub repos that match the strategy (e.g., `ai-infra-platform-engineer`, `ai-infra-platform-engineer-solutions`).
   - For separate solutions repos, plan automation (Actions, scheduled sync job) now.

4. **Initialize workspaces**
   ```bash
   mkdir -p research curriculum/{roles,platform-engineer,mlops-engineer}
   cp templates/research/role-research-template.md research/platform-engineer/role-research.md
   cp templates/research/job-posting-analysis-template.md research/platform-engineer/job-posting-analysis.md
   # …repeat for each role
   cp templates/curriculum/master-plan-template.yaml curriculum/platform-engineer/master-plan.yaml
   cp templates/curriculum/module-roadmap-template.md curriculum/platform-engineer/modules/module-01-roadmap.md
   cp templates/curriculum/project-plan-template.md curriculum/platform-engineer/projects/project-01-plan.md
   cp templates/curriculum/multi-role-alignment-template.md curriculum/roles/multi-role-alignment.md
   ```

5. **Set up collaboration**
   - Review `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, and `SECURITY.md`.
   - Configure GitHub issue templates (already included) and Discussions for Q&A.

---

## Phase 1 – Role Research & Analysis

Follow `workflows/multi-role-program.md` (Phase 1) for each target role:

1. **Run AI prompts for drafts**
   - Use `prompts/research/role-research-prompt.md` with role-specific context.
   - Store human-edited output in `research/<role>/role-research.md`.

2. **Job posting analysis**
   - Collect ≥20 postings, record stats in `job-posting-analysis.md`.
   - Use `rg "TODO"` to ensure no placeholders remain.

3. **Practitioner interviews**
   - Capture summaries using `templates/research/practitioner-interview-template.md`.
   - Link audio/video or transcripts in the template metadata.

4. **Skills matrix synthesis**
   - Summarize evidence (postings + interviews + market trends).
   - Use `prompts/research/skills-matrix-prompt.md` to generate an initial YAML matrix.
   - Validate with SMEs, ensure progression levels reference evidence IDs (`JP-01`, `INT-02`).

5. **Cross-role coordination**
   - Update `curriculum/roles/multi-role-alignment.md` (Progression Ladder, Role Comparison Matrix) as each role is researched.
   - Identify shared competencies early to reduce duplication later.

Deliverables checklist:
- `[ ]` Role research brief per role
- `[ ]` Job posting analysis per role
- `[ ]` ≥5 practitioner interviews per role
- `[ ]` Skills matrix YAML per role
- `[ ]` Multi-role alignment dashboard initialised

---

## Phase 2 – Curriculum Planning

Use `workflows/curriculum-design.md` with research inputs:

1. **Master plan**
   - Populate `curriculum/<role>/master-plan.yaml` with learning outcomes, module list, assessment strategy, solution plan summary.
   - Ensure `repository_config` points to `curriculum/repository-strategy.yaml`.

2. **Module roadmaps**
   - For each module:
     - Copy `templates/curriculum/module-roadmap-template.md`.
     - Fill cross-role progression, solutions plan, quality checklist.
     - Map learning objectives to competency levels from the skills matrix.

3. **Project plans**
   - Use `templates/curriculum/project-plan-template.md` for each anchor/stretches project.
   - Document reuse strategy (what advanced roles inherit or extend).

4. **Multi-role alignment**
   - Update `curriculum/roles/multi-role-alignment.md` with module assignment per role, shared assets, differentiators.
   - Confirm modules build sequentially across roles (avoid rewriting similar content).

5. **Repository strategy iteration**
   - Adjust `curriculum/repository-strategy.yaml` with real module/project IDs and paths.
   - Capture automation owners and review cadence for shared components.

Deliverables checklist:
- `[ ]` Master plan per role
- `[ ]` Module roadmaps for every module
- `[ ]` Project plans for anchor + stretch projects
- `[ ]` Updated multi-role alignment dashboard
- `[ ]` Repository strategy file with concrete mappings

---

## Phase 3 – Learning Content Generation

Use `workflows/module-generation.md` per module. Recommended approach:

1. **Module specification**
   - Confirm prerequisites, word counts, topics from module roadmap.
   - Use the prompts in `workflows/module-generation.md` step-by-step.

2. **AI-assisted drafting**
   - Use `prompts/lecture-generation/comprehensive-module-prompt.md`.
   - Chunk generation by sections to manage token length (Introduction, core concepts, advanced topics, etc.).

3. **Enhancement**
   - Enrich code examples using `prompts/code-generation/production-code-examples-prompt.md`.
   - Ensure case studies cite public sources (conference talks, blog posts).

4. **Exercises & assessments**
   - Duplicate `templates/exercises/exercise-template.md` and customize.
   - Use `templates/assessments/quiz-assessment-template.md` to draft quizzes.
   - Document each asset in the module roadmap’s Practical Components table.

5. **Validation**
   - Run automated checks:
     ```bash
     python validation/completeness/check-module-completeness.py path/to/module
     python validation/code-validators/validate-code-examples.py lecture-notes.md
     ```
   - Follow `validation/content-checkers/module-quality-checklist.md`.

6. **Documentation packaging**
   - Create module README summarizing contents.
   - Update multi-role dashboard with module status (draft/in review/complete).

Deliverables:
- `[ ]` Lecture notes (≥12,000 words, with production-quality code and case studies)
- `[ ]` Exercises (5–10 per module) and solutions placeholders
- `[ ]` Assessment (quiz, rubric)
- `[ ]` Module README + meta files
- `[ ]` Validation reports archived (e.g., `validation/module-<id>-report.md`)

---

## Phase 4 – Project Development

With module content drafted, follow `workflows/project-generation.md`:

1. **Starter repository**
   - Scaffold `starter/` directory with TODO-marked code and documentation.
   - Align with module(s) prerequisites from project plan.

2. **Implementation guide**
   - Build `IMPLEMENTATION_GUIDE.md` per workflow instructions.
   - Ensure troubleshooting and validation sections reference module learnings.

3. **Assessment material**
   - Define rubric in `ASSESSMENT.md`, aligned with competencies.
   - Include cross-role variants or extensions.

4. **Integrations**
   - Document how projects align with earlier/later modules.
   - For multi-role progression, note upgrade paths in project plan.

5. **Validation**
   - Run tests (`pytest`, `npm test`, etc.) and record results.
   - Ensure TODOs in starter code are explicit and tracked.

Deliverables:
- `[ ]` Project directory with starter code, implementation guide, assessment
- `[ ]` Project README (overview, setup, resources)
- `[ ]` Validation logs (tests, linting, security scans)

---

## Phase 5 – Solutions Production

Follow the solutions process in `docs/architecture.md` (Phase 5) and `workflows/module-generation.md` Phase 5:

1. **Confirm repo placement**
   - Use `curriculum/repository-strategy.yaml` to identify repo/path.
   - If separate repo, create and secure (private/instructor-only).

2. **Exercise solutions**
   - For each exercise, copy `templates/solutions/exercise-solution-template.md`.
   - Provide step-by-step resolution, validation commands, troubleshooting notes.

3. **Project solutions**
   - Implement production-grade solution per project plan.
   - Document using `templates/solutions/project-solution-template.md`.
   - Record validation matrix (tests, security scans, performance).

4. **Assessment solutions**
   - Draft answer keys and rubrics with `templates/solutions/assessment-solution-template.md`.
   - Highlight common mistakes and feedback snippets.

5. **Cross-role reuse**
   - Note shared libraries or infra modules in solutions templates.
   - Update multi-role dashboard’s “Shared Assets” section.

6. **Access control**
   - Apply repo permissions (e.g., instructors group).
   - Document release cadence and sync automation.

Deliverables:
- `[ ]` Solutions folders populated per asset type
- `[ ]` Validation evidence captured in templates
- `[ ]` Repo permissions configured
- `[ ]` Multi-role dashboard updated with reuse notes

---

## Phase 6 – Quality Assurance & Release

1. **Automated validation**
   - Run validation suite across content & solutions:
     ```bash
     python validation/code-validators/validate-code-examples.py path/to/lecture-notes.md
     python validation/completeness/check-module-completeness.py path/to/module
     # Project-specific checks (pytest, linting, bandit, etc.)
     ```

2. **Manual review**
   - Peer review modules, exercises, assessments, and solutions.
   - Ensure cross-role alignment (no duplicated text; advanced roles build on earlier ones).

3. **Security review**
   - Follow `SECURITY.md` best practices: scan dependencies, confirm secrets handled appropriately, etc.

4. **Changelog & release notes**
   - Update `CHANGELOG.md` with additions per release.
   - Create `RELEASE.md` or GitHub Release describing assets and status (draft/pilot/final).

5. **Publishing**
   - Push learning and solutions repos per strategy.
   - Tag release (e.g., `v0.3.0-ai-infra-curriculum`).
   - Communicate via Discussions or mailing list as appropriate.

Deliverables:
- `[ ]` QA report with resolved issues
- `[ ]` Updated changelog & release notes
- `[ ]` Published repositories/tags per strategy
- `[ ]` Announcement/discussion thread (optional)

---

## Operational Tips

- **Issue tracking**: Use the GitHub issue templates (bugs, features, documentation, templates/workflows) to coordinate workstreams.
- **Automation**: Set up GitHub Actions to run validation scripts on PRs and solutions repos.
- **Metrics**: Track progress in `curriculum/roles/multi-role-alignment.md` (e.g., status columns per module/project).
- **Review cadence**: Align with governance plan in `templates/curriculum/repository-strategy-template.yaml` (quarterly reviews, approvers).
- **Continuous improvement**: After each module/project release, document lessons learned in `CHANGELOG.md` or a dedicated `retrospectives/` folder.

---

By following this guide, a small core team—or a coordinated set of AI agents with human oversight—can produce the entire AI Infrastructure Curriculum, including research foundations, robust learning materials, validated hands-on projects, and secured solution repositories. Use this as a playbook to drive consistent, high-quality curriculum production end to end.
