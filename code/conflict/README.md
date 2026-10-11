# Conflict Track Placeholder

This directory is reserved for the DRIFT-Bench conflict track.

The conflict track should evaluate whether a fusion model can:

1. expose source-grounded factual conflicts,
2. distinguish genuine conflicts from omissions or surface variation,
3. resolve addressed conflicts faithfully, and
4. report conflict-handling behavior through auditable structured outputs.

## Planned Layout

```text
fusion/
  # conflict-aware fusion runners or conflict-injection fusion runners
alignment/
  # alignment and conflict-list grounding utilities
evaluation/
  # conflict exposure and resolution-faithfulness audits
score/
  # metric aggregation, tables, and figures
```

## Expected Inputs

```text
../../data/conflict/
  rewritten_or_injected/
  conflict_lists/
```

## Expected Outputs

```text
../../outputs/conflict/
  fused_news/
  alignment/
  eval/
  score/
```

## Reserved Metrics

Future conflict-track scoring should preserve the paper-level concepts:

- Stage 1 conflict exposure: explicitly reported, silently handled, or unaddressed.
- Stage 2 resolution faithfulness: majority success, logical neutral, reconciliation hallucination, or propagation failure.
- Declaration precision for spurious reported conflicts.

The current public release keeps this branch as a stable placeholder; the implemented runnable pipeline is under `../non_conflict/`.
