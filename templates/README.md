# Templates Overview

This directory contains reusable building blocks for the AI Infrastructure Content Generator.

## Structure

| Folder | Purpose |
|--------|---------|
| `lecture-notes/` | Long-form module templates |
| `exercises/` | Guided practice templates |
| `projects/` | Hands-on project scaffolds |
| `assessments/` | Quiz and assessment templates |
| `research/` | Market research and analysis templates |
| `curriculum/` | Program planning artifacts (master plans, roadmaps, repo strategy) |
| `solutions/` | Solution packages for exercises, projects, assessments |
| `partials/` | Optional sections that can be included inside other templates |

## Using Partials

Partials are modular snippets that can be included in other markdown templates.  
We recommend the `!INCLUDE path/to/partial` convention or your preferred static-site include syntax.

Example include:

```markdown
<!-- inside a lecture note template -->
!INCLUDE templates/partials/production-best-practices.md
```

Feel free to create new partials for shared sections (e.g., observability guidance, governance checklists).
