# Conflict Track

This directory contains the runnable DRIFT-Bench conflict-track pipeline. The
scripts use the canonical configuration in `../common/` and are grouped by
pipeline stage.

## Layout

```text
fusion/
  fusion_system_conflict.py
  fusion_system_conflict_zh.py
alignment/
  conflict.py
  aligner.py
evaluation/
  evaluate_conflict.py
  evaluate_conflict_zh.py
score/
  # metric aggregation, tables, and figures
```

`alignment/conflict.py` injects controlled entity-level conflicts, while
`alignment/aligner.py` aligns fused reports with their source variants.

## Data and output locations

Keep inputs under `../../data/conflict/` and generated artifacts under
`../../outputs/conflict/`. API credentials are read from environment variables
as described in the repository root README; no credentials are stored here.

The long-running alignment and evaluation scripts write JSONL checkpoints next
to their requested output files. This allows interrupted jobs to resume without
reprocessing completed records. Treat files under `outputs/conflict/` as
generated artifacts rather than source code.

## Typical workflow

Run commands from the repository root.

### 1. Inject controlled conflicts

```bash
python code/conflict/alignment/conflict.py \
  --input data/conflict/rewritten/source.json \
  --output data/conflict/rewritten_or_injected/injected.jsonl \
  --model gpt-4o-mini
```

### 2. Fuse and expose conflicts

Use `fusion_system_conflict.py` for English and
`fusion_system_conflict_zh.py` for Chinese.

```bash
python code/conflict/fusion/fusion_system_conflict.py \
  --input data/conflict/rewritten_or_injected/injected.json \
  --output outputs/conflict/fused_news/fused_en.json \
  --model gpt-4o \
  --fusion-method direct
```

### 3. Align the fused report

```bash
python code/conflict/alignment/aligner.py \
  --input outputs/conflict/fused_news/fused_en.json \
  --output outputs/conflict/alignment/aligned_en.json \
  --model gpt-5.4 \
  --workers 4
```

### 4. Evaluate conflict handling

Use `evaluate_conflict.py` for English and `evaluate_conflict_zh.py` for
Chinese.

```bash
python code/conflict/evaluation/evaluate_conflict.py \
  --input outputs/conflict/alignment/aligned_en.json \
  --output outputs/conflict/eval/evaluated_en.json \
  --model gpt-5.4 \
  --workers 4
```

The evaluation records conflict exposure and resolution-faithfulness outcomes
in structured output for downstream scoring.

## Shared configuration

Model definitions and runtime defaults live in `../common/`:

```text
../common/config.py
../common/config_api.py
../common/config_models.py
../common/config_runtime.py
```

Set `DRIFT_API_KEY_1` and, when needed, `DRIFT_API_BASE_1` before calling a
hosted model. Local OpenAI-compatible endpoints can be configured in
`../common/config_models.py`. The similarity model location can be overridden
with `DRIFT_SIMILARITY_MODEL_PATH`.
