# Exporters

This directory contains configuration stubs and guidance for publishing generated content to downstream destinations (documentation portals, LMS platforms, static sites, etc.).

## Export Targets

| Config | Description |
|--------|-------------|
| `samples/mkdocs.yml` | Example MkDocs build configuration for publishing modules as a documentation site. |
| `samples/lms-export.json` | Mapping of modules/projects to LMS identifiers for bulk uploads. |

## Recommended Workflow

1. Maintain asset metadata with paths to generated content (`metadata.yaml` files validated via `schemas/asset-metadata.schema.json`).
2. Configure exporters to map repository content to the target format.
3. Use the CLI helper to bootstrap navigation or mapping files:
   ```bash
   ./tools/curriculum.py generate-mkdocs-nav modules/ > exporters/mkdocs-nav.yaml
   ```
4. Integrate with CI/CD (e.g., GitHub Actions) to render/publish on demand.

## Examples

### MkDocs

```yaml
site_name: "AI Infrastructure Curriculum"
nav:
  - Home: index.md
  - Modules:
      - "Module 01": modules/module-01/lesson.md
      - "Module 02": modules/module-02/lesson.md
markdown_extensions:
  - admonition
  - toc:
      permalink: true
```

### LMS Export Mapping

```json
{
  "course_id": "mlops-specialist",
  "modules": [
    { "id": "MOD-01", "path": "modules/module-01/lesson.md", "lms_module_id": "mlops-101" },
    { "id": "MOD-02", "path": "modules/module-02/lesson.md", "lms_module_id": "mlops-201" }
  ]
}
```

## Next Steps

- Add automation (scripts/Actions) that consume these files and push to the destination.
- Extend the CLI (`tools/curriculum.py`) with commands to generate exporter configs from metadata.
- Document actual exporter usage in `docs/metadata-and-automation.md` once integrated.
