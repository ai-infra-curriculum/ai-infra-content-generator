# Schemas

This directory contains JSON schemas used by the framework.

## Available Schemas

| Schema | Description |
|--------|-------------|
| `asset-metadata.schema.json` | Validates metadata for modules, exercises, projects, assessments, and solutions. |

## Validating Metadata

Use the CLI helper:

```bash
./tools/curriculum.py validate-metadata path/to/metadata.yaml
```

The command uses `jsonschema` to report validation errors with field-level context.

## Extending Schemas

- Add new fields as the framework evolves (e.g., exporters, review workflows).
- Keep `metadata_version` constant until a breaking change is introduced.
- Document schema updates in `CHANGELOG.md`.
