# AI Infrastructure Content Generator

A system-agnostic framework for generating comprehensive technical curriculum content using AI assistance.

---

## ⚠️ AI-Generated Content Disclaimer

> **Important Notice**: This repository contains tools and frameworks for generating AI-assisted educational content. The generated content is designed to undergo human review and verification. While we strive for accuracy, **generated content may contain errors, inaccuracies, or outdated information**. 
>
> **Status**: 🔄 Active development
>
> Please use this framework responsibly:
> - Review all generated content before publication
> - Validate code examples and technical accuracy
> - Cross-reference with official documentation
> - Test in safe environments
>
> We appreciate your understanding as we develop tools for responsible AI-assisted content creation.

---

## Overview

This repository provides a clean, system-agnostic framework for generating comprehensive technical educational content. It extracts successful patterns and methodologies from the AI Infrastructure Curriculum project and packages them as reusable tools.

## Purpose

The goal is to create high-quality, production-ready curriculum materials including:
- Comprehensive lecture notes (12,000+ words)
- Hands-on exercises and projects
- Real-world case studies
- Code examples and implementations
- Assessment materials

## Key Features

- **System Agnostic**: Works with any LLM or AI system
- **Template Driven**: Reusable templates for consistent content
- **Quality Focused**: Built-in quality checks and validation
- **Scalable**: Generate content for multiple modules in parallel
- **Documented**: Clear processes and best practices
- **Checkpoint System**: Save and resume work across sessions with automated progress tracking
- **Research Toolkit**: Standardized prompts and workbooks for market research and skills mapping
- **Multi-Role Ready**: Dedicated workflow and dashboards to coordinate curricula across roles
- **Configurable Solutions Delivery**: Plan inline or standalone solution repositories with automated templates

## Repository Structure

```
ai-infra-content-generator/
├── README.md                    # This file
├── docs/                        # Documentation
│   ├── getting-started.md      # Quick start guide (8,300 words)
│   ├── architecture.md         # System design, 8-phase workflow (12,500 words)
│   ├── best-practices.md       # Content generation best practices (15,400 words)
│   ├── tools-and-automation.md # 30 MCP servers, 6 skills, automation (20,500 words)
│   ├── agent-playbook.md       # 8-phase agent orchestration (10,800 words)
│   ├── metadata-and-automation.md # Metadata schema, pipelines, CLI usage
│   ├── micro-learning-guide.md   # Micro-learning sprint workflow
│   └── ai-infrastructure-curriculum-guide.md # End-to-end curriculum production playbook
├── configs/                     # Validation profiles and automation settings
│   └── validation-profiles.yaml
├── pipelines/                   # Declarative workflow manifests
│   ├── single-module.yaml
│   ├── micro-learning.yaml
│   └── ai-infra-program.yaml
├── schemas/                     # JSON schemas for metadata/validation
│   └── asset-metadata.schema.json
├── graphs/                      # GraphQL schema for cross-role mapping
│   └── ai-infra.graphql
├── tools/                       # CLI helpers and automation scripts
│   └── curriculum.py
├── exporters/                   # Publishing configs for docs/LMS
│   └── README.md
├── templates/                   # Content templates
│   ├── research/               # Role briefs, job analysis, interviews, skills matrix
│   ├── curriculum/             # Master plan, module roadmaps, project plans, multi-role dashboard
│   ├── solutions/              # Solution packages for exercises/projects/assessments
│   ├── lecture-notes/          # Lecture note templates
│   ├── exercises/              # Exercise templates
│   ├── projects/               # Project templates
│   ├── assessments/            # Quiz and test templates
│   ├── partials/               # Optional sections for reuse
│   ├── module-metadata-template.yaml
│   ├── project-metadata-template.yaml
│   └── README.md               # Template usage guide
├── prompts/                     # AI prompts and instructions
│   ├── research/               # Prompts for role and skills research
│   ├── lecture-generation/     # Prompts for lecture content
│   ├── code-generation/        # Prompts for code examples
│   ├── solutions/              # Prompts for solution artifacts
│   ├── case-studies/           # Prompts for real-world examples
│   ├── README.md               # Prompt usage & versioning
│   └── version-map.yaml        # Prompt version history
├── validation/                  # Quality validation tools
│   ├── content-checkers/       # Content quality checks
│   ├── code-validators/        # Code validation
│   └── completeness/           # Completeness checks
├── workflows/                   # Generation workflows
│   ├── module-generation.md    # How to generate a module
│   ├── project-generation.md   # How to generate projects
│   ├── curriculum-design.md    # Curriculum design process
│   └── multi-role-program.md   # Coordinating research & curriculum across roles
└── examples/                    # Example outputs
    ├── sample-module/          # Complete module example
    └── sample-project/         # Complete project example
└── lessons/                     # Micro-learning samples
    └── sample-micro/           # Micro-lesson example
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ai-infra-curriculum/ai-infra-content-generator.git
   cd ai-infra-content-generator
   ```

2. **Review the documentation**
   ```bash
   cat docs/getting-started.md
   ```

3. **Explore templates**
   ```bash
   ls templates/
   ```

4. **Generate your first module**
   Follow the workflow in `workflows/module-generation.md`

5. **Plan the complete program**
   Walk through the end-to-end playbook in `docs/ai-infrastructure-curriculum-guide.md`

6. **Install optional tooling helpers**
   ```bash
   pip install -r requirements-tooling.txt
   ./tools/curriculum.py pipelines
   ./tools/curriculum.py export-graph examples/sample-module examples/sample-project lessons/sample-micro
   ```

7. **Review sample assets**
   - Module + exercise: `examples/sample-module/`
   - Project brief: `examples/sample-project/`
   - Micro-lesson: `lessons/sample-micro/`

## Research & Curriculum Planning Starter

- Copy research templates for each target role: `cp templates/research/* research/<role-slug>/`
- Use `prompts/research/role-research-prompt.md` to draft role briefs, then validate with job postings and interviews
- Build curriculum plans with `templates/curriculum/master-plan-template.yaml` and `templates/curriculum/module-roadmap-template.md`
- Coordinate overlapping roles through `workflows/multi-role-program.md` and the `templates/curriculum/multi-role-alignment-template.md` dashboard
- Configure repository and solutions strategy via `templates/curriculum/repository-strategy-template.yaml` and align module roadmaps accordingly
- Generate metadata stubs per asset using `templates/curriculum/*-metadata-template.yaml` or `./tools/curriculum.py scaffold-metadata ...`, then validate with `./tools/curriculum.py validate-metadata`.

## Solutions & Repository Configuration

- Decide whether solutions ship with the main curriculum (`solutions.placement: inline`) or in dedicated repositories (`solutions.placement: separate`).
- Choose a single shared repository or one per role using `repositories.mode`.
- Generate solution artifacts with `templates/solutions/*` and `prompts/solutions/solution-generation-prompt.md`.
- Use `workflows/module-generation.md` and `workflows/project-generation.md` to publish solutions without duplicating content across roles.

## Automation Toolkit

- Pipeline manifests (`pipelines/*.yaml`) describe reusable workflows. Inspect them with `./tools/curriculum.py pipelines`.
- Available manifests include `single-module`, `micro-learning`, and `ai-infra-program`.
- Asset metadata should conform to `schemas/asset-metadata.schema.json`; validate via `./tools/curriculum.py validate-metadata path/to/metadata.yaml`.
- Reuse validation bundles defined in `configs/validation-profiles.yaml` (`./tools/curriculum.py validation-profiles`).
- Execute validation commands or scaffold metadata with `./tools/curriculum.py run-validation <profile> <path>` and `./tools/curriculum.py scaffold-metadata ...`.
- Export graph data for dashboards using `./tools/curriculum.py export-graph modules/ projects/ --output graphs/generated.json`.
- Cross-role relationships can be modeled with `graphs/ai-infra.graphql` for dashboards or LMS exports.
- Template partials in `templates/partials/` let you add or swap content blocks without cloning entire templates.
- Exporter configs in `exporters/` help publish to documentation portals or LMS platforms (`./tools/curriculum.py generate-mkdocs-nav modules/`).
- Sample GitHub workflow (`.github/workflows/validation.yml`) demonstrates automated metadata validation and docs checks.

## What Makes This Different?

Based on successful generation of 36,000+ words of technical content and expanded with multi-role program support:

- ✅ **Proven Patterns**: Extracted from successful MLOps curriculum generation
- ✅ **Quality Standards**: 12,000+ word modules with comprehensive coverage
- ✅ **Real-World Focus**: Includes case studies and production examples
- ✅ **Validation Built-In**: Quality checks and completeness verification
- ✅ **Scalable Process**: Generate single modules or entire multi-role programs
- ✅ **Research Framework**: Standardized templates for market research and skills analysis
- ✅ **Multi-Role Ready**: Coordinate curriculum across multiple job roles efficiently

**Framework Stats**:
- 📄 ≈200 files (templates, workflows, prompts, tools)
- 📝 50,000+ words of documentation and guidance
- 🎯 15,000+ words of core documentation
- 🛠️ 2 automated validation scripts
- 🔧 30 documented MCP servers
- 👥 Multi-role program orchestration support

## Use Cases

- Creating technical training curricula
- Generating comprehensive documentation
- Building educational content repositories
- Developing certification programs
- Creating learning paths for engineering roles

## Philosophy

1. **AI-Assisted, Human-Reviewed**: AI generates comprehensive drafts, humans verify accuracy
2. **Quality Over Speed**: 12,000+ word modules ensure depth and completeness
3. **Production-Ready**: All examples are tested and production-grade
4. **Transparency**: Clear disclaimers about AI-generation and verification status
5. **Continuous Improvement**: Learn from each generation cycle

## Community

We're building a community around AI-assisted educational content creation:

- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards and guidelines
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to this project
- **[Support](SUPPORT.md)** - Get help, troubleshooting, and resources
- **[Security Policy](SECURITY.md)** - Report vulnerabilities and security concerns
- **[Changelog](CHANGELOG.md)** - Version history and release notes

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Improving documentation
- Creating templates and prompts
- Building validation tools
- Sharing workflows and examples

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Additional Notice**: Content generated using this framework should be reviewed by qualified subject matter experts before publication. The framework authors are not responsible for the accuracy or appropriateness of generated content. See LICENSE for full terms.

## Credits

Developed as part of the AI Infrastructure Curriculum project.

## Contact

- GitHub: https://github.com/ai-infra-curriculum
- Issues: https://github.com/ai-infra-curriculum/ai-infra-content-generator/issues

---

**Status**: 🚧 Under active development

**Version**: 0.2.0 (Multi-role program support added)
