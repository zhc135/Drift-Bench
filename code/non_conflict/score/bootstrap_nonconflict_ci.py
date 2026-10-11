#!/usr/bin/env python3
"""
Bootstrap confidence intervals for summary scoring metrics (S_logic, S_lang, S_info, S_fact, S_total).

Supports:
- Per-model CIs (existing)
- Overall CI: pool all models' news units and bootstrap the pooled set
- Macro-average CI: bootstrap each model independently, average scores across models

Outputs CSV, Markdown, and LaTeX tables.
"""

from __future__ import annotations

import csv
import json
import math
import random
import os
import argparse
from typing import Dict, List, Tuple, Any, Optional

# ==================== Configuration ====================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_ROOT = os.getenv("DRIFT_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs", "non_conflict"))
SCORE_DIR = os.getenv("DRIFT_SCORE_DIR", os.path.join(OUTPUT_ROOT, "score"))
TOTAL_NEWS = 150   # per model, after excluding id=003

# Model configurations (copy from original)
MODELS = {
    "qw7": {"path": "qw7", "display_name": "qwen-2.5-7b-instruct"},
    "gpt-4o": {"path": "gpt-4o", "display_name": "gpt-4o"},
    "deepseek-v3.1": {"path": "deepseek-v3.1", "display_name": "deepseek-v3.1"},
    "gemma": {"path": "gemma", "display_name": "gemma-3-12b-it"},
    "gpt-oss-20b": {"path": "gpt-oss-20b", "display_name": "gpt-oss-20b"},
    "llama": {"path": "llama", "display_name": "llama-3.1-8B-instruct"},
    "phi-4": {"path": "phi-4", "display_name": "phi-4"},
    "claude-sonnet-4-5": {"path": "claude-sonnet-4-5", "display_name": "claude-sonnet-4-5"},
    "gemini-2.5-flash": {"path": "gemini-2.5-flash", "display_name": "gemini-2.5-flash"},
}

# Lambda parameters
LAMBDA_INFO = math.log(2) / 2
LAMBDA_LANG = math.log(2) / 4
LAMBDA_LOGIC = math.log(2)

# Bootstrap settings (can be overridden by CLI)
N_BOOT = 5000
SEED = 20260711

METRIC_NAMES = ["S_logic", "S_lang", "S_info", "S_fact", "S_total"]
METRIC_LABELS = {
    "S_logic": "Logic Score",
    "S_lang": "Language Score",
    "S_info": "Info Score",
    "S_fact": "Factuality Score",
    "S_total": "Total Score",
}

# ==================== Scoring functions ====================

def calculate_category_score(total_problems: int, lambda_val: float, total_news: int) -> float:
    if total_problems == 0:
        return 100.0
    E_avg = total_problems / total_news
    return round(100.0 * math.exp(-lambda_val * E_avg), 2)

def calculate_total_score(logic_score: float, lang_score: float, info_score: float) -> float:
    if logic_score <= 0 or lang_score <= 0 or info_score <= 0:
        return 0.0
    return round(math.pow(logic_score * lang_score * info_score, 1/3), 2)

def calculate_fact_score(info_score: float, logic_score: float) -> float:
    if info_score <= 0 or logic_score <= 0:
        return 0.0
    return round(math.sqrt(info_score * logic_score), 2)

def compute_all_scores(logic_total: int, lang_total: int, info_total: int, total_news: int) -> Dict[str, float]:
    S_logic = calculate_category_score(logic_total, LAMBDA_LOGIC, total_news)
    S_lang  = calculate_category_score(lang_total,  LAMBDA_LANG,  total_news)
    S_info  = calculate_category_score(info_total,  LAMBDA_INFO,  total_news)
    S_fact  = calculate_fact_score(S_info, S_logic)
    S_total = calculate_total_score(S_logic, S_lang, S_info)
    return {
        "S_logic": S_logic,
        "S_lang": S_lang,
        "S_info": S_info,
        "S_fact": S_fact,
        "S_total": S_total,
    }

# ==================== Data loading ====================

def load_per_news_counts(score_dir: str, model_key: str, language: str) -> Optional[List[Dict[str, int]]]:
    base_name = f"{language}_{model_key}"
    logic_file = os.path.join(score_dir, model_key, f"{base_name}_logic.json")
    lang_file  = os.path.join(score_dir, model_key, f"{base_name}_lang.json")
    info_file  = os.path.join(score_dir, model_key, f"{base_name}_info.json")

    missing = []
    if not os.path.exists(logic_file): missing.append("logic")
    if not os.path.exists(lang_file):  missing.append("lang")
    if not os.path.exists(info_file):  missing.append("info")
    if missing:
        print(f"⚠️  Missing files for {model_key}/{language}: {missing}, skipping")
        return None

    with open(logic_file, 'r', encoding='utf-8') as f:
        logic_data = json.load(f)
    logic_scores = logic_data.get("scores", [])
    logic_counts = [sum(item.get("problem_counts", {}).values()) for item in logic_scores]

    with open(lang_file, 'r', encoding='utf-8') as f:
        lang_data = json.load(f)
    lang_scores = lang_data.get("scores", [])
    lang_counts = [item.get("problem_count", 0) for item in lang_scores]

    with open(info_file, 'r', encoding='utf-8') as f:
        info_data = json.load(f)
    info_per_news = info_data.get("per_news_scores", [])
    info_counts = [item.get("total_issues", 0) for item in info_per_news]

    if not (len(logic_counts) == len(lang_counts) == len(info_counts)):
        print(f"❌ Inconsistent lengths for {model_key}/{language}")
        return None

    return [{"logic": lc, "lang": lng, "info": ic}
            for lc, lng, ic in zip(logic_counts, lang_counts, info_counts)]

# ==================== Bootstrap helpers ====================

def percentile(data: List[float], q: float) -> float:
    if not data:
        return float("nan")
    pos = (len(data) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[int(pos)]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)

def bootstrap_scores_from_items(items: List[Dict[str, int]], total_news: int,
                                rng: random.Random) -> Dict[str, Tuple[float, float]]:
    """
    Generic bootstrap: given a list of per-news counts, resample with replacement
    (same length as items), compute scores, return 95% CI.
    """
    n = len(items)
    draws: Dict[str, List[float]] = {metric: [] for metric in METRIC_NAMES}
    for _ in range(N_BOOT):
        indices = [rng.randrange(n) for _ in range(n)]
        total_logic = sum(items[i]["logic"] for i in indices)
        total_lang  = sum(items[i]["lang"]  for i in indices)
        total_info  = sum(items[i]["info"]  for i in indices)
        scores = compute_all_scores(total_logic, total_lang, total_info, total_news)
        for metric in METRIC_NAMES:
            draws[metric].append(scores[metric])
    cis = {}
    for metric in METRIC_NAMES:
        vals = sorted(draws[metric])
        cis[metric] = (percentile(vals, 0.025), percentile(vals, 0.975))
    return cis

def bootstrap_macro_average(model_items: Dict[str, List[Dict[str, int]]],
                            total_news_per_model: int,
                            rng: random.Random) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap macro average: for each model, resample its own items, compute score,
    then average across models. Return CI of the average.
    """
    model_names = list(model_items.keys())
    draws: Dict[str, List[float]] = {metric: [] for metric in METRIC_NAMES}
    for _ in range(N_BOOT):
        model_scores = []
        for name in model_names:
            items = model_items[name]
            n = len(items)
            indices = [rng.randrange(n) for _ in range(n)]
            total_logic = sum(items[i]["logic"] for i in indices)
            total_lang  = sum(items[i]["lang"]  for i in indices)
            total_info  = sum(items[i]["info"]  for i in indices)
            scores = compute_all_scores(total_logic, total_lang, total_info, total_news_per_model)
            model_scores.append(scores)
        # Average across models
        avg_scores = {}
        for metric in METRIC_NAMES:
            vals = [s[metric] for s in model_scores]
            avg_scores[metric] = sum(vals) / len(vals) if vals else float("nan")
        for metric in METRIC_NAMES:
            draws[metric].append(avg_scores[metric])
    cis = {}
    for metric in METRIC_NAMES:
        vals = sorted(draws[metric])
        cis[metric] = (percentile(vals, 0.025), percentile(vals, 0.975))
    return cis

# ==================== Main processing ====================

def main():
    global N_BOOT, SEED

    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for summary scores")
    parser.add_argument("--score-dir", type=str, default=SCORE_DIR)
    parser.add_argument("--total-news", type=int, default=TOTAL_NEWS,
                        help="Number of news per model (after filtering)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--output-csv", type=str, default="bootstrap_ci_summary_scores.csv")
    parser.add_argument("--output-md", type=str, default="bootstrap_ci_summary_scores.md")
    parser.add_argument("--output-tex", type=str, default="bootstrap_ci_summary_scores.tex")
    args = parser.parse_args()

    N_BOOT = args.n_boot
    SEED = args.seed
    rng = random.Random(SEED)

    # Load all model data per language
    lang_data: Dict[str, Dict[str, List[Dict[str, int]]]] = {}
    for lang in ["zh", "en"]:
        lang_data[lang] = {}
        for model_key in MODELS:
            items = load_per_news_counts(args.score_dir, model_key, lang)
            if items is not None:
                lang_data[lang][model_key] = items
        if not lang_data[lang]:
            print(f"⚠️  No data loaded for language {lang}, skipping")
            continue

    # Prepare results structure: language -> model -> {point, ci, n_news}
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {lang: {} for lang in lang_data}

    for lang, model_items in lang_data.items():
        # 1. Per-model CIs
        for model, items in model_items.items():
            point = compute_all_scores(
                sum(i["logic"] for i in items),
                sum(i["lang"]  for i in items),
                sum(i["info"]  for i in items),
                args.total_news
            )
            ci = bootstrap_scores_from_items(items, args.total_news, rng)
            all_results[lang][model] = {
                "point": point,
                "ci": ci,
                "n_news": len(items)
            }

        # 2. Overall: pool all items from all models
        pooled_items = []
        for items in model_items.values():
            pooled_items.extend(items)
        total_news_pooled = len(pooled_items)   # = N_models * args.total_news
        point_pooled = compute_all_scores(
            sum(i["logic"] for i in pooled_items),
            sum(i["lang"]  for i in pooled_items),
            sum(i["info"]  for i in pooled_items),
            total_news_pooled
        )
        ci_pooled = bootstrap_scores_from_items(pooled_items, total_news_pooled, rng)
        all_results[lang]["Overall"] = {
            "point": point_pooled,
            "ci": ci_pooled,
            "n_news": total_news_pooled,
        }

        # 3. Macro average: average of model scores after bootstrap
        ci_macro = bootstrap_macro_average(model_items, args.total_news, rng)
        # Point estimate for macro: compute each model's score on original data, then average
        model_points = []
        for items in model_items.values():
            p = compute_all_scores(
                sum(i["logic"] for i in items),
                sum(i["lang"]  for i in items),
                sum(i["info"]  for i in items),
                args.total_news
            )
            model_points.append(p)
        point_macro = {}
        for metric in METRIC_NAMES:
            vals = [p[metric] for p in model_points]
            point_macro[metric] = sum(vals) / len(vals) if vals else float("nan")
        all_results[lang]["Macro Avg"] = {
            "point": point_macro,
            "ci": ci_macro,
            "n_news": args.total_news,  # not really meaningful for macro
        }

    # Write outputs
    write_csv_results(all_results, args.output_csv)
    write_markdown_summary(all_results, args.output_md)
    write_tex_summary(all_results, args.output_tex)

    print(f"\n✅ Results written to:")
    print(f"  - {args.output_csv}")
    print(f"  - {args.output_md}")
    print(f"  - {args.output_tex}")

# ==================== Output writers ====================

def write_csv_results(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_path: str) -> None:
    fields = ["language", "model", "metric", "point_estimate", "ci_low", "ci_high", "n_news", "n_boot"]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for lang in sorted(all_results.keys()):
            for model in sorted(all_results[lang].keys()):
                res = all_results[lang][model]
                point = res["point"]
                ci = res["ci"]
                n_news = res.get("n_news", "")
                for metric in METRIC_NAMES:
                    writer.writerow({
                        "language": lang,
                        "model": model,
                        "metric": METRIC_LABELS.get(metric, metric),
                        "point_estimate": round(point[metric], 2),
                        "ci_low": round(ci[metric][0], 2),
                        "ci_high": round(ci[metric][1], 2),
                        "n_news": n_news,
                        "n_boot": N_BOOT,
                    })

def write_markdown_summary(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Bootstrap Confidence Intervals for Summary Scores\n\n")
        f.write(f"**Bootstrap resamples:** {N_BOOT}, **Seed:** {SEED}\n\n")
        for lang in sorted(all_results.keys()):
            f.write(f"## Language: {lang.upper()}\n\n")
            f.write("| Model | " + " | ".join(METRIC_LABELS[m] for m in METRIC_NAMES) + " |\n")
            f.write("|-------|" + "|".join(["------"] * len(METRIC_NAMES)) + "|\n")
            # Ensure "Overall" and "Macro Avg" appear at bottom
            models = sorted(all_results[lang].keys(), key=lambda x: (x not in ["Overall", "Macro Avg"], x))
            for model in models:
                res = all_results[lang][model]
                point = res["point"]
                ci = res["ci"]
                row = [model]
                for metric in METRIC_NAMES:
                    val = point[metric]
                    low, high = ci[metric]
                    row.append(f"{val:.2f} [{low:.2f}, {high:.2f}]")
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

def write_tex_summary(all_results: Dict[str, Dict[str, Dict[str, Any]]], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated by bootstrap_summary_scores.py\n")
        f.write("\\begin{tabular}{l" + "c" * len(METRIC_NAMES) + "}\n")
        f.write("\\toprule\n")
        f.write("Model & " + " & ".join(METRIC_LABELS[m] for m in METRIC_NAMES) + " \\\\\n")
        f.write("\\midrule\n")
        for lang in sorted(all_results.keys()):
            f.write(f"\\multicolumn{{{len(METRIC_NAMES)+1}}}{{c}}{{\\textbf{{{lang.upper()}}}}} \\\\\n")
            models = sorted(all_results[lang].keys(), key=lambda x: (x not in ["Overall", "Macro Avg"], x))
            for model in models:
                res = all_results[lang][model]
                point = res["point"]
                ci = res["ci"]
                cells = [model]
                for metric in METRIC_NAMES:
                    val = point[metric]
                    low, high = ci[metric]
                    cells.append(f"{val:.2f} [{low:.2f}, {high:.2f}]")
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

if __name__ == "__main__":
    main()
