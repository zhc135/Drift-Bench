# Release Manifest

This manifest records the current public repository scope. The runnable pipeline is the non-conflict track. The conflict track is represented by a stable directory branch and file-layout contract for future code.

## Public Path Convention

```text
data/non_conflict/
data/conflict/
outputs/non_conflict/
outputs/conflict/
models/
```

Scripts also accept environment-variable overrides:

```text
DRIFT_DATA_ROOT
DRIFT_OUTPUT_ROOT
DRIFT_MODELS_ROOT
DRIFT_SIMILARITY_MODEL_PATH
```

## Common Configuration

Canonical shared configuration:

```text
code/common/config.py
code/common/config_api.py
code/common/config_models.py
code/common/config_runtime.py
```

The non-conflict scripts currently keep local config copies under `code/non_conflict/` for backward-compatible imports.

## Non-Conflict Track

Included code:

```text
code/non_conflict/rewrite_system.py
code/non_conflict/extract_re.py
code/non_conflict/tools/clean_punct.py
code/non_conflict/fusion_system.py
code/non_conflict/fusion_system_zh.py
code/non_conflict/aligner.py
code/non_conflict/info.py
code/non_conflict/info_zh.py
code/non_conflict/lang.py
code/non_conflict/lang_zh.py
code/non_conflict/logic.py
code/non_conflict/logic_zh.py
code/non_conflict/score/auto.py
code/non_conflict/score/bootstrap_nonconflict_ci.py
code/non_conflict/score/draw1.py
code/non_conflict/draw_compare.py
code/non_conflict/draw_auto_metric.py
code/non_conflict/analyze_length_buckets.py
code/non_conflict/logic_analysis.py
code/non_conflict/evaluate_fusion.py
```

Expected inputs:

```text
data/non_conflict/classified_disaster_news/*.json
data/non_conflict/all_rewritten/summary_*.json
data/non_conflict/fusion_eval/en.json
data/non_conflict/fusion_eval/zh.json
```

`data/non_conflict/fusion_eval/en.json` and `data/non_conflict/fusion_eval/zh.json` contain the complete non-conflict English and Chinese examples used for the full fusion and evaluation experiments.

Expected outputs:

```text
outputs/non_conflict/fused_news/fused/fused_<lang>_<model>.json
outputs/non_conflict/fused_news/origin/aligned_<lang>_<model>.json
outputs/non_conflict/eval/<model>/<lang>_<model>_<dimension>_gpt-oss-120b.json
outputs/non_conflict/score/<model>/<lang>_<model>_{info,lang,logic}.json
outputs/non_conflict/bootstrap_ci_summary_scores.{csv,md,tex}
```

Rationale:

- English fusion uses `fusion_system.py`.
- Chinese fusion uses `fusion_system_zh.py`.
- Final non-conflict alignment uses `aligner.py`.
- English language-quality evaluation uses `lang.py`; `la.py` was treated as an older candidate and is not included.
- Chinese evaluation scripts use the `_zh` suffix.
- `score/auto.py` uses the final nine-model set and excludes `id == "003"` before scoring 150 events per language/model.

## Conflict Track

Reserved branch:

```text
code/conflict/
  README.md
  fusion/
  alignment/
  evaluation/
  score/
data/conflict/
outputs/conflict/
```

Planned outputs:

```text
outputs/conflict/fused_news/
outputs/conflict/alignment/
outputs/conflict/eval/
outputs/conflict/score/
```

The current public release does not yet include runnable conflict-track scripts. The reserved branch keeps the future public API stable.

## Excluded Legacy/Internal Items

```text
align_test.py
aligner_conflict.py
conflict.py
fusion_system_conflict.py
fusion_enhanced.py
human_judge/
fused_news/origin_unfixed/
sco/
scored/
ev/
internal scratch README files
extracted paper text files
```

Reasons:

- Conflict-track code is not yet prepared for the public branch.
- Human-judgment utilities are separate from the automated non-conflict pipeline.
- Old or intermediate result directories can conflict with the final reported results.
- Internal scratch documentation and extracted paper text are not suitable for a public source release.
