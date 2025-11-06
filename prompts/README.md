# Prompt Library

Prompts are organized by use case (research, lecture generation, solutions, etc.).  
To support versioning, each prompt may include a header with `Prompt-Version`. Consumers can pin to a version or adopt `latest`.

## Versioning

- Update `version-map.yaml` when making substantive prompt changes.
- Include a `Prompt-Version: vX.Y.Z` line at the top of the prompt.
- Document changes in `CHANGELOG.md` if they affect outputs materially.

## Categories

| Directory | Focus |
|-----------|-------|
| `research/` | Role research, skills synthesis |
| `lecture-generation/` | Lecture module drafting |
| `code-generation/` | Code examples and production readiness |
| `solutions/` | Solution artifact generation |
| `case-studies/` | Real-world case study prompts |

## Suggested Workflow

1. Choose prompts based on pipeline manifest instructions.
2. Fill bracketed placeholders with context from research and curriculum plans.
3. Track prompt versions used in asset metadata (`prompts_used` field).
