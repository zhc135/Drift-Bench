# DRIFT-Bench

DRIFT-Bench is a benchmark project for faithful and transparent disaster report information fusion. The repository is organized around two tracks:

- **Non-Conflict Track**: multi-source disaster news fusion when source variants are fragmented but not intentionally contradictory.
- **Conflict Track**: multi-source disaster news fusion when one or more controlled factual conflicts are injected and the model must expose and resolve them transparently.

Both the non-conflict pipeline and the conflict-track injection, fusion,
alignment, and evaluation scripts are organized under `code/`. See
`code/conflict/README.md` for the conflict workflow.

## Repository Layout

```text
code/
  common/
    config.py
    config_api.py
    config_models.py
    config_runtime.py
  non_conflict/
    rewrite_system.py
    extract_re.py
    fusion_system.py
    fusion_system_zh.py
    aligner.py
    info.py
    info_zh.py
    lang.py
    lang_zh.py
    logic.py
    logic_zh.py
    evaluate_fusion.py
    draw_auto_metric.py
    draw_compare.py
    analyze_length_buckets.py
    logic_analysis.py
    score/
      auto.py
      bootstrap_nonconflict_ci.py
      draw1.py
    tools/
      clean_punct.py
  conflict/
    README.md
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
data/
  non_conflict/
  conflict/
outputs/
  non_conflict/
  conflict/
models/
docs/
  USED_FILES.md
```

`code/common/` is the canonical place for shared configuration. The current non-conflict scripts also keep local copies of the config files for backward-compatible imports; a later package refactor can remove that duplication.

## Setup

```bash
cd drift_bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Third-party API keys are not included. Set your own key before running scripts that call hosted models:

```bash
export DRIFT_API_KEY_1="your_api_key"
export DRIFT_API_BASE_1="https://api.openai.com/v1"
```

For local vLLM-served models, edit the model endpoints in `code/common/config_models.py` and `code/non_conflict/config_models.py`, or launch compatible OpenAI-style endpoints on the configured `localhost:<port>` URLs.

Useful path overrides:

```bash
export DRIFT_DATA_ROOT="$PWD/data"
export DRIFT_OUTPUT_ROOT="$PWD/outputs/non_conflict"
export DRIFT_MODELS_ROOT="$PWD/models"
export DRIFT_SIMILARITY_MODEL_PATH="$PWD/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Data Layout

Recommended public layout:

```text
data/
  non_conflict/
    classified_disaster_news/
      earthquake.json
      flood.json
      public_health.json
      finance.json
      turmoil.json
      en1.json
    all_rewritten/
      summary_earthquake.json
      summary_flood.json
      summary_public_health.json
      summary_finance.json
      summary_turmoil.json
      summary_other.json
      summary_en.json
  conflict/
    README.md or released conflict-track files
outputs/
  non_conflict/
    fused_news/fused/
    fused_news/origin/
    eval/
    score/
  conflict/
    fused_news/
    alignment/
    eval/
    score/
```

The prepared project directory includes the non-conflict input data and the corresponding fused, aligned, evaluation, score, and analysis outputs under `data/non_conflict/` and `outputs/non_conflict/`. Conflict-track code is available under `code/conflict/`; conflict data and generated outputs belong under `data/conflict/` and `outputs/conflict/`.

The non-conflict track evaluates 300 events: 150 English and 150 Chinese. Some intermediate fused files may contain 151 entries; the scoring script excludes `id == "003"` and scores 150 events per model/language pair.

## Non-Conflict Track

Run the following commands from `code/non_conflict/`.

```bash
cd code/non_conflict
```

### 1. Rewrite and Collect Source Variants

```bash
python rewrite_system.py \
  -i ../../data/non_conflict/classified_disaster_news/flood.json \
  -o ../../outputs/non_conflict/rewritten_files/flood_outputs
```

Collect individual `news_*_summary.json` files into one summary file:

```bash
python extract_re.py
```

When prompted, enter the directory containing `news_*_summary.json`. The script writes `summary_collection.json` in that directory.

Optional punctuation cleanup:

```bash
python tools/clean_punct.py \
  ../../data/non_conflict/all_rewritten/summary_finance.json \
  ../../data/non_conflict/all_rewritten/summary_finance.cleaned.json
```

### 2. Fuse Rewritten Reports

English fusion uses `fusion_system.py`; Chinese fusion uses `fusion_system_zh.py`. The final non-conflict experiments use direct fusion.

English example:

```bash
python fusion_system.py \
  --input ../../data/non_conflict/all_rewritten/summary_en.json \
  --output ../../outputs/non_conflict/fused_news/fused/fused_en_gpt-4o.json \
  --model gpt-4o \
  --fusion-method direct
```

Chinese example:

```bash
python fusion_system_zh.py \
  --input ../../data/non_conflict/all_rewritten/summary_flood.json \
  --output ../../outputs/non_conflict/fused_news/fused/fused_zh_gpt-4o.json \
  --model gpt-4o \
  --fusion-method direct
```

Models used in the non-conflict table:

```text
claude-sonnet-4-5
deepseek-v3.1
gemini-2.5-flash
gpt-4o
gpt-oss-20b
gemma-3-12b-it
llama3.1-8b-Instruct
phi-4
qwen_7b / qw7
```

### 3. Align Fused Text With Source Variants

```bash
python aligner.py \
  --input ../../outputs/non_conflict/fused_news/fused/fused_en_gpt-4o.json \
  --output ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  --model gpt-5.4 \
  --workers 4
```

Repair failed entries in an existing alignment file:

```bash
python aligner.py \
  --input ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  --output ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  --model gpt-5.4 \
  --fix-mode
```

### 4. TRACE Non-Conflict Evaluation

The non-conflict evaluation uses three dimensions:

- Information Completeness: `info.py` / `info_zh.py`
- Language Quality: `lang.py` / `lang_zh.py`
- Logical Coherence: `logic.py` / `logic_zh.py`

The paper experiments used `gpt-oss-120b` as the non-conflict judge.

Information completeness:

```bash
python info.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/en_gpt-4o_info_gpt-oss-120b.json \
  -sm gpt-oss-120b

python info_zh.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_zh_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/zh_gpt-4o_info_gpt-oss-120b.json \
  -sm gpt-oss-120b
```

Language quality:

```bash
python lang.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/en_gpt-4o_language_quality_gpt-oss-120b.json \
  -sm gpt-oss-120b

python lang_zh.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_zh_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/zh_gpt-4o_language_quality_gpt-oss-120b.json \
  -sm gpt-oss-120b
```

Logical coherence:

```bash
python logic.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_en_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/en_gpt-4o_logic_gpt-oss-120b.json \
  -sm gpt-oss-120b

python logic_zh.py \
  -i ../../outputs/non_conflict/fused_news/origin/aligned_zh_gpt-4o.json \
  -o ../../outputs/non_conflict/eval/gpt-4o/zh_gpt-4o_logic_gpt-oss-120b.json \
  -sm gpt-oss-120b
```

Repair an existing evaluation file:

```bash
python lang.py \
  -i ../../outputs/non_conflict/eval/gpt-4o/en_gpt-4o_language_quality_gpt-oss-120b.json \
  --fix-mode \
  -sm gpt-oss-120b
```

### 5. Score and Analyze

```bash
python score/auto.py \
  --eval-dir ../../outputs/non_conflict/eval \
  --score-dir ../../outputs/non_conflict/score \
  --total-news 150
```

Generate bootstrap confidence intervals:

```bash
python score/bootstrap_nonconflict_ci.py \
  --score-dir ../../outputs/non_conflict/score \
  --total-news 150 \
  --output-csv ../../outputs/non_conflict/bootstrap_ci_summary_scores.csv \
  --output-md ../../outputs/non_conflict/bootstrap_ci_summary_scores.md \
  --output-tex ../../outputs/non_conflict/bootstrap_ci_summary_scores.tex
```

Additional plotting and analysis scripts:

```bash
python draw_compare.py
python draw_auto_metric.py
python analyze_length_buckets.py
python logic_analysis.py
python score/draw1.py
```

## Conflict Track

The conflict track is organized under:

```text
code/conflict/
data/conflict/
outputs/conflict/
```

Implemented submodules:

```text
code/conflict/fusion/       # conflict-aware fusion runners
code/conflict/alignment/    # conflict injection and source alignment
code/conflict/evaluation/   # exposure and resolution-faithfulness audits
code/conflict/score/        # conflict metrics and tables
```

The conflict-track release adds the following runnable components:

- `alignment/conflict.py`: injects controlled entity-level factual conflicts
  into rewritten source variants.
- `fusion/fusion_system_conflict.py`: performs English conflict-aware fusion
  and emits a structured conflict summary.
- `fusion/fusion_system_conflict_zh.py`: Chinese counterpart of the
  conflict-aware fusion runner.
- `alignment/aligner.py`: aligns fused statements with their supporting source
  variants so conflict decisions remain auditable.
- `evaluation/evaluate_conflict.py`: evaluates English conflict exposure and
  resolution behavior.
- `evaluation/evaluate_conflict_zh.py`: evaluates the same dimensions for
  Chinese outputs.

These scripts load shared model, API, and runtime settings from `code/common/`.
Hosted-model credentials must be supplied through the environment variables
described in [Setup](#setup); API keys are not stored in the repository.

The end-to-end order is:

```text
rewritten variants
  -> controlled conflict injection
  -> conflict-aware fusion
  -> source alignment
  -> conflict evaluation
  -> score aggregation
```

Output layout:

```text
outputs/conflict/
  fused_news/
  alignment/
  eval/
  score/
```

The README in `code/conflict/` documents the runnable workflow and command examples.

## Reproducing From an Existing Private Data Directory

If you already have an internal data/output directory, point the public code at it with environment variables instead of editing source files:

```bash
export DRIFT_DATA_ROOT="/path/to/data"
export DRIFT_OUTPUT_ROOT="/path/to/outputs/non_conflict"
python code/non_conflict/score/auto.py
```

For `score/auto.py`, `DRIFT_OUTPUT_ROOT` should contain `eval/`.

## Release Notes

- API keys are intentionally excluded; configure them through environment variables.
- Large data, generated outputs, and local model checkpoints are ignored by default.
- Conflict-track code is not yet included, but the directory branch and output contract are reserved.
