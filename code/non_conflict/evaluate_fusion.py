#!/usr/bin/env python3
# coding: utf-8

"""
评估融合结果：对每个 fused_*.json 文件，
- 可选择剔除指定 ID 的数据（例如 id='003'）
- 每条数据：计算生成文本 fused_content 与 rewritten_v1/v2/v3 的三个指标平均值（指标1），
          以及 fused_content 与 input_text 的指标（指标2）。
- 文件级别：计算所有剩余数据条目的指标1平均值和指标2平均值。
- 汇总所有文件的结果到一个 CSV/JSON 文件。
- 输出各文件（模型）的指标对比表格。
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer

def tokenize_char(text: str) -> List[str]:
    """字符级分词：保留每个汉字、字母、数字、标点符号作为一个 token。"""
    text = text.strip()
    return [ch for ch in text if not ch.isspace()]

def tokenize_bleu(text: str) -> List[str]:
    """BLEU 使用的分词，简单采用字符级。"""
    return tokenize_char(text)

def prepare_for_rouge(text: str) -> str:
    """将分词结果用空格连接，供 rouge_score 使用。"""
    tokens = tokenize_char(text)
    return ' '.join(tokens)

def compute_rouge_scores(hyp: str, ref: str, rouge_types: List[str] = None) -> Dict[str, float]:
    """计算 ROUGE 分数，返回指定类型的 F1 值。"""
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
    hyp_prep = prepare_for_rouge(hyp)
    ref_prep = prepare_for_rouge(ref)
    scores = scorer.score(ref_prep, hyp_prep)
    result = {}
    for key in rouge_types:
        result[key] = scores[key].fmeasure
    return result

def compute_bleu_score(hyp: str, ref: str) -> float:
    """计算 BLEU-4 分数（使用平滑）。"""
    hyp_tokens = tokenize_bleu(hyp)
    ref_tokens = tokenize_bleu(ref)
    smoothing = SmoothingFunction().method4
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return bleu

def compute_bertscore_single(hyp: str, ref: str, bert_scorer: BERTScorer) -> float:
    """利用已初始化的 BERTScorer 计算单个对的 F1 分数。"""
    P, R, F1 = bert_scorer.score([hyp], [ref])
    return F1.item()

def batch_compute_bertscore(hyp_list: List[str], ref_list: List[str], bert_scorer: BERTScorer) -> List[float]:
    """批量计算 BERTScore F1。"""
    P, R, F1 = bert_scorer.score(hyp_list, ref_list)
    return F1.tolist()

def evaluate_item(fused_text: str, ref_texts: List[str], bert_scorer: BERTScorer) -> Dict[str, Any]:
    """
    评估一条数据：计算 fused_text 与每个参考文本的指标，返回各指标的平均值（针对多个参考）。
    如果 ref_texts 为空，返回 None。
    """
    if not fused_text or not ref_texts:
        return None

    rouge1_list, rouge2_list, rougeL_list = [], [], []
    bleu_list = []
    hyp_list = []
    ref_list_for_bert = []
    for ref in ref_texts:
        if not ref:
            continue
        rouge_scores = compute_rouge_scores(fused_text, ref)
        rouge1_list.append(rouge_scores['rouge1'])
        rouge2_list.append(rouge_scores['rouge2'])
        rougeL_list.append(rouge_scores['rougeL'])
        bleu_list.append(compute_bleu_score(fused_text, ref))
        hyp_list.append(fused_text)
        ref_list_for_bert.append(ref)

    if not rouge1_list:
        return None

    bert_f1_list = batch_compute_bertscore(hyp_list, ref_list_for_bert, bert_scorer)

    avg_rouge1 = np.mean(rouge1_list)
    avg_rouge2 = np.mean(rouge2_list)
    avg_rougeL = np.mean(rougeL_list)
    avg_bleu = np.mean(bleu_list)
    avg_bertscore = np.mean(bert_f1_list)

    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'bleu': avg_bleu,
        'bertscore': avg_bertscore
    }

def process_one_file(file_path: str, bert_scorer: BERTScorer, exclude_id: Optional[str] = None) -> Tuple[Dict[str, float], List[Dict]]:
    """
    处理单个 fused JSON 文件。
    如果指定 exclude_id，则剔除 id 等于该值的所有数据条目。
    返回 (文件级别平均指标, 每条记录的详细指标列表)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    item_results = []
    file_metrics = {
        'rouge1_ref3avg': [], 'rouge2_ref3avg': [], 'rougeL_ref3avg': [], 'bleu_ref3avg': [], 'bertscore_ref3avg': [],
        'rouge1_ref_input': [], 'rouge2_ref_input': [], 'rougeL_ref_input': [], 'bleu_ref_input': [], 'bertscore_ref_input': []
    }

    for item in data:
        # 剔除指定 ID 的数据
        item_id = str(item.get('id', ''))
        if exclude_id and item_id == exclude_id:
            print(f"  Excluding item with id={exclude_id} from {os.path.basename(file_path)}")
            continue

        fused = item.get('fused_content', '').strip()
        if not fused:
            continue

        ref_versions = []
        for v in ['rewritten_v1', 'rewritten_v2', 'rewritten_v3']:
            txt = item.get(v, '').strip()
            if txt:
                ref_versions.append(txt)
        input_text = item.get('input_text', '').strip()

        metrics1 = evaluate_item(fused, ref_versions, bert_scorer) if ref_versions else None
        metrics2 = evaluate_item(fused, [input_text], bert_scorer) if input_text else None

        record = {
            'id': item_id,
            'metrics_v1_v2_v3_avg': metrics1,
            'metrics_input_text': metrics2
        }
        item_results.append(record)

        if metrics1:
            for k in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bertscore']:
                file_metrics[f'{k}_ref3avg'].append(metrics1[k])
        if metrics2:
            for k in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bertscore']:
                file_metrics[f'{k}_ref_input'].append(metrics2[k])

    file_avg = {}
    for key, values in file_metrics.items():
        if values:
            file_avg[key] = float(np.mean(values))
        else:
            file_avg[key] = 0.0

    return file_avg, item_results

def print_comparison_table(summaries: Dict[str, Dict[str, float]], metric_groups: List[str], title: str):
    """
    打印模型（文件）之间的指标对比表格。
    summaries: {模型名: {指标名: 值}}
    metric_groups: 要展示的指标列表，例如 ['rouge1_ref3avg', 'rouge2_ref3avg', ...]
    """
    if not summaries:
        print("No data to display.")
        return

    # 表头
    headers = ["Model"] + metric_groups
    # 计算每列最大宽度（至少为表头长度）
    col_widths = [len(h) for h in headers]
    model_names = list(summaries.keys())
    for name in model_names:
        col_widths[0] = max(col_widths[0], len(name))
    for i, metric in enumerate(metric_groups, start=1):
        for name in model_names:
            val = summaries[name].get(metric, 0.0)
            col_widths[i] = max(col_widths[i], len(f"{val:.4f}"))
        # 表头本身也考虑
        col_widths[i] = max(col_widths[i], len(metric))

    # 分隔线
    sep = "+".join("-" * (w + 2) for w in col_widths)
    sep = "+" + sep + "+"

    def format_row(cells):
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, col_widths)) + " |"

    print(f"\n{title}")
    print(sep)
    print(format_row(headers))
    print(sep)
    for name in model_names:
        row = [name]
        for metric in metric_groups:
            val = summaries[name].get(metric, 0.0)
            row.append(f"{val:.4f}")
        print(format_row(row))
    print(sep)

def main():
    parser = argparse.ArgumentParser(description='Evaluate fused news files with ROUGE, BLEU, BERTScore.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing fused_*.json files')
    parser.add_argument('--output_summary', type=str, default='evaluation_summary.json',
                        help='Output JSON file for file-level summary')
    parser.add_argument('--output_details', type=str, default='',
                        help='Optional output JSON file for per-item details (if empty, not saved)')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='Pretrained model for BERTScore (e.g., bert-base-chinese, roberta-large)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for BERTScore computation')
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                        help='Device for BERTScore (cuda or cpu)')
    parser.add_argument('--exclude_id', type=str, default='003',
                        help='Exclude data entries with this id (e.g., "003"). Set to empty string to disable.')
    args = parser.parse_args()

    # 初始化 BERTScorer
    print(f"Loading BERTScorer with model {args.bert_model} on {args.device}...")
    bert_scorer = BERTScorer(model_type=args.bert_model, lang='zh', device=args.device,
                             batch_size=args.batch_size, rescale_with_baseline=True)

    pattern = os.path.join(args.input_dir, 'fused_*.json')
    file_paths = glob.glob(pattern)
    if not file_paths:
        print(f"No files found matching {pattern}")
        return

    print(f"Found {len(file_paths)} files to process.")
    if args.exclude_id:
        print(f"Will exclude entries with id = '{args.exclude_id}'")

    all_file_summaries = {}
    all_details = {}

    for fp in tqdm(file_paths, desc="Processing files"):
        base = os.path.basename(fp)
        if base.startswith('fused_'):
            key = base[6:-5]  # 去掉 'fused_' 和 '.json'
        else:
            key = base[:-5]
        print(f"\nProcessing {base} -> {key}")

        file_avg, item_details = process_one_file(fp, bert_scorer, exclude_id=args.exclude_id if args.exclude_id else None)
        all_file_summaries[key] = file_avg
        if args.output_details:
            all_details[key] = item_details

    # 保存文件级别的汇总
    with open(args.output_summary, 'w', encoding='utf-8') as f:
        json.dump(all_file_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nFile-level summary saved to {args.output_summary}")

    if args.output_details:
        with open(args.output_details, 'w', encoding='utf-8') as f:
            json.dump(all_details, f, ensure_ascii=False, indent=2)
        print(f"Per-item details saved to {args.output_details}")

    # 打印所有文件的整体平均（剔除后）
    print("\n=== Overall average across all files (after exclusion) ===")
    overall = {}
    for metric in ['rouge1_ref3avg', 'rouge2_ref3avg', 'rougeL_ref3avg', 'bleu_ref3avg', 'bertscore_ref3avg',
                   'rouge1_ref_input', 'rouge2_ref_input', 'rougeL_ref_input', 'bleu_ref_input', 'bertscore_ref_input']:
        values = [summ[metric] for summ in all_file_summaries.values() if summ.get(metric, 0) > 0]
        if values:
            overall[metric] = np.mean(values)
        else:
            overall[metric] = 0.0
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    # 打印每个模型（文件）的指标对比表格
    if all_file_summaries:
        # 定义两组指标
        metrics_vs_refs = ['rouge1_ref3avg', 'rouge2_ref3avg', 'rougeL_ref3avg', 'bleu_ref3avg', 'bertscore_ref3avg']
        metrics_vs_input = ['rouge1_ref_input', 'rouge2_ref_input', 'rougeL_ref_input', 'bleu_ref_input', 'bertscore_ref_input']

        print_comparison_table(all_file_summaries, metrics_vs_refs,
                               "Table 1: Metrics averaged over rewritten_v1/v2/v3 (higher is better)")
        print_comparison_table(all_file_summaries, metrics_vs_input,
                               "Table 2: Metrics compared with input_text (higher is better)")

if __name__ == '__main__':
    main()