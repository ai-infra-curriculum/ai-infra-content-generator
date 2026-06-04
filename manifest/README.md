# Curriculum manifest + canonical-source registry

Two structured indexes that ground every AICG content decision.

## Files

| File | Purpose | Updated by |
|---|---|---|
| `curriculum.manifest.json` | Every track, module, exercise, project, and resource across the 27 curriculum repos, with paths + GitHub URLs | `scripts/build-curriculum-manifest.py` (run after any structural curriculum change) |
| `canonical_sources.json` | For each external tool/concept the curriculum cites: canonical URL + known successors (vendor rebrand, deprecation, OCI migration, etc.) | **human-curated** — committed by hand when you learn about a vendor change |

## Who reads them

| Consumer | File | Why |
|---|---|---|
| `aicg/link_refresh.py` resolver | `canonical_sources.json` | Routes a deprecated vendor URL to the live successor BEFORE falling back to Wayback. Also enforces "always machine-consumed" protection for endpoints like `fulcio.sigstore.dev` |
| `aicg/research.py` agent prompt builder | `curriculum.manifest.json` (via `summarize_manifest_for_prompt`) | Gives the research agent ground truth about what's already covered, so it proposes net-new content instead of duplicating |
| Moderation bot (`/find`, `/ask`) | `curriculum.manifest.json` | Cross-reference lookups by track / module / exercise slug |
| Steward / weekly-audit jobs | both | Detects manifest drift (modules referenced in cross-references that no longer exist; canonical entries whose `successors` are themselves now broken) |

## Rebuilding the curriculum manifest

```sh
python scripts/build-curriculum-manifest.py --workspace ~/path/to/parent-of-repos
# defaults to workspace=parent-of-CWD, output=manifest/curriculum.manifest.json
```

Idempotent — re-run anytime. Reproduces the same output for an unchanged workspace.

Run it after any of these events to keep the manifest in sync:

- A new module / exercise / project lands in a `-learning` or `-solutions` repo
- A research proposal is merged
- A reorganization touches `lessons/` ↔ `modules/` ↔ `projects/` paths

It's cheap (~1 second for the whole workspace) so a wrapper like `scripts/refresh-curriculum-manifest.sh` can be safely called from any pipeline step that mutates curriculum structure.

## Editing canonical_sources.json

Add a new entry when you discover that the curriculum is citing a URL whose canonical form is somewhere else now. Shape:

```json
{
  "name": "Short human label (vendor + topic)",
  "canonical": "https://canonical-url-or-oci://...",
  "topic": ["tag1", "tag2"],
  "rationale": "One sentence: what happened to the original, what's authoritative now.",
  "machine_consumed": false,   // true if the URL is consumed by tooling (helm/oci/cosign), so Wayback must NEVER substitute even if 404
  "successors": {
    "https://old-url-that-was-cited": "https://or-oci-canonical-url",
    "https://another-old-cited-variant": "https://same-canonical"
  }
}
```

The resolver matches both `url` and `url.rstrip("/")` against `successors`, so you don't need separate entries for trailing-slash variants.

## Schema versions

Both files carry `schema_version: 1`. Bump it (and the constant in `src/aicg/manifest.py`) when you make a backwards-incompatible change to the structure. The loader rejects mismatched versions to prevent silent drift.
