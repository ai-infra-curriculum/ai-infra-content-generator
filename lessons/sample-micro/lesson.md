# Micro-Lesson: Feature Flag Rollout Checklist

## Objective

Prepare a safe rollout plan for introducing a new feature flag across an existing service.

## Checklist

1. **Flag Design**
   - Determine default state (`off`).
   - Scope blast radius (subset of traffic, region, or tenant).
2. **Implementation**
   - Add guard clauses with clear naming.
   - Emit logs when the flag is evaluated.
3. **Observability**
   - Create dashboards segmented by flag state.
   - Set alerts on error rate and latency when flag == on.
4. **Rollout Plan**
   - Enable for internal users → beta customers → 25% traffic → 100%.
   - Define rollback steps (toggle off, invalidate caches).
5. **Post-Rollout**
   - Capture learnings.
   - Schedule flag removal when feature stabilizes.

## Quick Exercise

Draft a rollout message for stakeholders explaining the timeline, metrics to watch, and rollback decision criteria.
