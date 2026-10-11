#!/usr/bin/env python3
"""
按融合文本长度分桶统计平均问题数（中英文分别等频分桶 + 可视化）
改进点：
- 分桶策略：对中/英文数据分别采用等频分桶（每个桶样本量接近），避免数据过度集中
- 可视化：为每个模型生成折线图，并在一张图中左右并排显示中英文模型对比
- 输出：CSV 统计表 + PDF 图片

用法示例：

    # 等频分桶（默认每语言6桶），自动生成图片
    python analyze_length_buckets_visual.py \
        --input /path/to/dir1 /path/to/dir2 ... \
        --n-bins 6 --plot

    # 自定义等宽分桶（仍按语言分别独立计算边界）
    python analyze_length_buckets_visual.py \
        --input ... --method width --step 2000 --max-bin 35000 --plot

    # 只统计，不绘图
    python analyze_length_buckets_visual.py --input ... --output-dir ./results
"""

import json
import argparse
import sys
import csv
import math
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# 可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.2)
        sns.set_palette("colorblind")
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
        plt.style.use('seaborn-v0_8-whitegrid')
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("警告: matplotlib 未安装，无法生成图表。请执行 pip install matplotlib", file=sys.stderr)

# ---------------------------- 模型名称映射 ----------------------------
def get_model_display_name(basename: str) -> str:
    """根据文件名前缀映射到友好的模型名称"""
    mapping = {
        'claude-sonnet-4-5': 'Claude Sonnet 4.5',
        'deepseek-v3.1': 'DeepSeek-V3.1',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gpt-4o': 'GPT-4o',
        'gpt-oss-20b': 'GPT-OSS-20B',
        'phi-4': 'Phi-4',
        'gemma-3-12b-it': 'Gemma-3-12B-IT',
        'llama3.1-8b-instruct': 'Llama-3.1-8B',
        'qw7': 'Qwen-2.5-7B'
    }
    lower_name = basename.lower()
    for key, display in mapping.items():
        if key in lower_name:
            return display
    return basename

# ---------------------------- 文件查找 ----------------------------
def find_matching_files(paths: List[str], pattern: str = r".*(en|zh).*info.*\.json") -> List[str]:
    matched = []
    regex = re.compile(pattern, re.IGNORECASE)
    for p in paths:
        if os.path.isfile(p):
            if regex.search(os.path.basename(p)):
                matched.append(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.lower().endswith('.json') and regex.search(f):
                        matched.append(os.path.join(root, f))
        else:
            print(f"警告: 路径不存在或无法访问: {p}", file=sys.stderr)
    return matched

# ---------------------------- 数据加载 ----------------------------
def load_news_list(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    else:
        raise ValueError(f"文件 {file_path} 格式不符合预期（需要列表或包含data列表的对象）")

def get_fused_length(news: Dict[str, Any]) -> int:
    fused = news.get("fused_content", "")
    return len(str(fused)) if fused else 0

def get_problem_counts(news: Dict[str, Any]) -> Dict[str, int]:
    global_cnt = news.get("global_problem_count")
    para_cnt = news.get("paragraph_problem_count")
    if global_cnt is not None and para_cnt is not None:
        total = news.get("total_problem_count", global_cnt + para_cnt)
        return {"global": global_cnt, "paragraph": para_cnt, "total": total}
    dual = news.get("dual_evaluation", {})
    global_cnt = dual.get("global_evaluation", {}).get("total_global_problems", 0)
    para_cnt = dual.get("paragraph_evaluation", {}).get("total_paragraph_problems", 0)
    return {"global": global_cnt, "paragraph": para_cnt, "total": global_cnt + para_cnt}

def load_samples_by_language(file_paths: List[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, List[int]]]:
    """返回：{语言: [样本列表]}, {语言: [所有长度]}"""
    lang_samples = defaultdict(list)
    lang_lengths = defaultdict(list)
    for fp in file_paths:
        try:
            news_list = load_news_list(fp)
            base = os.path.basename(fp)
            lang = 'en' if 'en' in base.lower() else 'zh' if 'zh' in base.lower() else 'unknown'
            samples = []
            for news in news_list:
                length = get_fused_length(news)
                counts = get_problem_counts(news)
                samples.append({
                    "length": length,
                    "global": counts["global"],
                    "paragraph": counts["paragraph"],
                    "total": counts["total"]
                })
                lang_lengths[lang].append(length)
            lang_samples[lang].append((fp, samples))
            print(f"已加载 {len(samples)} 条新闻 ({lang}): {fp}", file=sys.stderr)
        except Exception as e:
            print(f"加载失败 {fp}: {e}", file=sys.stderr)
    return lang_samples, lang_lengths

# ---------------------------- 分桶边界生成 ----------------------------
def create_quantile_bins(lengths: List[int], n_bins: int) -> List[Tuple[int, int]]:
    if not lengths:
        return []
    data_sorted = sorted(lengths)
    n = len(data_sorted)
    percentiles = [int(100 * i / n_bins) for i in range(n_bins + 1)]
    percentiles[-1] = 100
    edges = []
    for p in percentiles:
        idx = int(math.ceil(p / 100.0 * n)) - 1
        idx = max(0, min(idx, n-1))
        val = data_sorted[idx]
        if not edges or val != edges[-1]:
            edges.append(val)
    if edges[-1] < max(lengths):
        edges.append(max(lengths) + 1)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

def create_width_bins(lengths: List[int], step: int, max_bound: Optional[int] = None) -> List[Tuple[int, int]]:
    max_len = max(lengths) if lengths else 0
    if max_bound is None:
        upper = ((max_len + step - 1) // step) * step
    else:
        upper = max_bound
    edges = list(range(0, upper + step, step))
    if edges[-1] < max_len:
        edges.append(max_len + 1)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

def create_log_bins(lengths: List[int], base: float = 2, min_bin: int = 500, max_bound: Optional[int] = None) -> List[Tuple[int, int]]:
    max_len = max(lengths) if lengths else 0
    upper = max_bound if max_bound is not None else (max_len + 1)
    edges = [0]
    val = min_bin
    while val < upper:
        edges.append(val)
        val = int(val * base)
    edges.append(upper)
    edges = sorted(set(edges))
    if edges[-1] < max_len:
        edges.append(max_len + 1)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

def create_custom_bins(bins_str: str, lengths: List[int]) -> List[Tuple[int, int]]:
    edges = [int(x.strip()) for x in bins_str.split(",")]
    max_len = max(lengths) if lengths else 0
    bins = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    if bins and bins[-1][1] <= max_len:
        bins[-1] = (bins[-1][0], max_len + 1)
    return bins

def get_bins_for_language(lengths: List[int], args, lang: str) -> List[Tuple[int, int]]:
    if args.method == "custom":
        if not args.bins:
            raise ValueError(f"custom 方法需要提供 --bins 参数")
        return create_custom_bins(args.bins, lengths)
    elif args.method == "width":
        return create_width_bins(lengths, args.step, args.max_bin)
    elif args.method == "quantile":
        return create_quantile_bins(lengths, args.n_bins)
    elif args.method == "log":
        return create_log_bins(lengths, args.base, args.min_bin, args.max_bin)
    else:
        raise ValueError(f"未知分桶方法: {args.method}")

def bin_label(low: int, high: int) -> str:
    return f"[{low}, {high})" if high > low else f"[{low}]"

# ---------------------------- 统计 ----------------------------
def compute_stats_for_file(samples: List[Dict], bins: List[Tuple[int, int]]) -> List[Dict]:
    stats = {}
    for low, high in bins:
        label = bin_label(low, high)
        stats[label] = {
            "count": 0, "sum_global": 0, "sum_paragraph": 0, "sum_total": 0,
            "low": low, "high": high
        }
    for s in samples:
        length = s["length"]
        assigned = False
        for low, high in bins:
            if low <= length < high:
                label = bin_label(low, high)
                stats[label]["count"] += 1
                stats[label]["sum_global"] += s["global"]
                stats[label]["sum_paragraph"] += s["paragraph"]
                stats[label]["sum_total"] += s["total"]
                assigned = True
                break
        if not assigned:
            last_label = bin_label(bins[-1][0], bins[-1][1])
            stats[last_label]["count"] += 1
            stats[last_label]["sum_global"] += s["global"]
            stats[last_label]["sum_paragraph"] += s["paragraph"]
            stats[last_label]["sum_total"] += s["total"]

    rows = []
    def label_key(label):
        m = re.search(r'\[(\d+)', label)
        return int(m.group(1)) if m else 0
    for label in sorted(stats.keys(), key=label_key):
        cnt = stats[label]["count"]
        avg_global = stats[label]["sum_global"] / cnt if cnt else 0
        avg_paragraph = stats[label]["sum_paragraph"] / cnt if cnt else 0
        avg_total = stats[label]["sum_total"] / cnt if cnt else 0
        rows.append({
            "长度区间": label,
            "新闻数量": cnt,
            "平均整体问题数": round(avg_global, 2),
            "平均段落问题数": round(avg_paragraph, 2),
            "平均总问题数": round(avg_total, 2),
            "区间中点": (stats[label]["low"] + stats[label]["high"]) / 2
        })
    return rows

def save_csv(rows: List[Dict], filepath: str):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["长度区间", "新闻数量", "平均整体问题数", "平均段落问题数", "平均总问题数"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"CSV 已保存: {filepath}", file=sys.stderr)

# ---------------------------- 单模型绘图（保存为 PDF） ----------------------------
def plot_model(rows: List[Dict], model_name: str, lang: str, output_dir: str):
    if not HAS_MPL or not rows:
        return
    data = [(r["区间中点"], r["平均总问题数"]) for r in rows if r["新闻数量"] > 0]
    if not data:
        return
    midpoints, avg_totals = zip(*data)
    plt.figure(figsize=(8, 5))
    plt.plot(midpoints, avg_totals, 
             marker='o', linestyle='-', linewidth=2, markersize=7,
             color='#d62728', markerfacecolor='#d62728', markeredgewidth=1, markeredgecolor='black',
             label=model_name)
    plt.xlabel("Fused Text Length (characters)", fontsize=12, fontweight='bold')
    plt.ylabel("Average Number of Problems", fontsize=12, fontweight='bold')
    title = f"{model_name} ({lang.upper()})\nAvg Total Problems vs. Fused Text Length"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6, color='gray')
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{model_name}_{lang}_bucket_plot.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {out_file}", file=sys.stderr)

# ---------------------------- 主控 ----------------------------
def main():
    parser = argparse.ArgumentParser(description="中英文分别等频分桶统计+可视化")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="输入文件或目录")
    parser.add_argument("--output-dir", "-o", default="./bucket_stats", help="输出目录")
    parser.add_argument("--method", choices=["width", "quantile", "log", "custom"], default="quantile",
                        help="分桶方法（默认quantile，即等频）")
    parser.add_argument("--bins", help="自定义边界（逗号分隔），method=custom时必须")
    parser.add_argument("--step", type=int, default=2000, help="等宽步长（默认2000）")
    parser.add_argument("--max-bin", type=int, default=35000, help="等宽或对数分桶的最大边界")
    parser.add_argument("--n-bins", type=int, default=6, help="等频分桶的桶数量（默认6）")
    parser.add_argument("--base", type=float, default=2, help="对数分桶底数")
    parser.add_argument("--min-bin", type=int, default=500, help="对数分桶最小边界")
    parser.add_argument("--pattern", default=r".*(en|zh).*info.*\.json", help="文件名匹配正则")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    parser.add_argument("--no-csv", action="store_true", help="不保存CSV")

    args = parser.parse_args()

    if args.plot and not HAS_MPL:
        print("错误: 需要安装 matplotlib", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 查找文件
    file_paths = find_matching_files(args.input, args.pattern)
    if not file_paths:
        print("错误: 未找到任何匹配的 JSON 文件", file=sys.stderr)
        sys.exit(1)
    print(f"找到 {len(file_paths)} 个文件待处理:", file=sys.stderr)

    # 2. 按语言加载样本和长度
    lang_samples, lang_lengths = load_samples_by_language(file_paths)
    if not lang_samples:
        print("错误: 没有成功加载任何数据", file=sys.stderr)
        sys.exit(1)

    # 3. 为每种语言生成分桶边界
    lang_bins = {}
    for lang, lengths in lang_lengths.items():
        if not lengths:
            continue
        print(f"\n语言 {lang.upper()} 样本数: {len(lengths)}, 长度范围: [{min(lengths)}, {max(lengths)}]", file=sys.stderr)
        try:
            bins = get_bins_for_language(lengths, args, lang)
            lang_bins[lang] = bins
            print(f"{lang.upper()} 分桶边界:", file=sys.stderr)
            for low, high in bins:
                print(f"  [{low}, {high})", file=sys.stderr)
        except Exception as e:
            print(f"生成 {lang} 分桶失败: {e}", file=sys.stderr)
            sys.exit(1)

    # 4. 对每个模型统计并输出
    all_model_data = []  # (lang, display_name, mids, avgs)
    for lang, file_samples_list in lang_samples.items():
        if lang not in lang_bins:
            continue
        bins = lang_bins[lang]
        for filepath, samples in file_samples_list:
            basename = os.path.basename(filepath).replace(".json", "")
            rows = compute_stats_for_file(samples, bins)

            # 终端输出
            print(f"\n===== {basename} (语言: {lang}) =====")
            print(f"{'长度区间':<20} {'新闻数量':<10} {'平均整体问题':<12} {'平均段落问题':<12} {'平均总问题':<10}")
            for r in rows:
                print(f"{r['长度区间']:<20} {r['新闻数量']:<10} {r['平均整体问题数']:<12} {r['平均段落问题数']:<12} {r['平均总问题数']:<10}")
            print("================================\n")

            if not args.no_csv:
                csv_path = os.path.join(args.output_dir, f"{basename}_bucket_stats.csv")
                save_csv(rows, csv_path)

            if args.plot:
                plot_model(rows, basename, lang, args.output_dir)

            # 收集用于对比图的数据
            plot_data = [(r["区间中点"], r["平均总问题数"]) for r in rows if r["新闻数量"] > 0]
            if plot_data:
                mids, avgs = zip(*plot_data)
                display_name = get_model_display_name(basename)
                all_model_data.append((lang, display_name, mids, avgs))

    # 5. 生成合并对比图（左右子图：英文 vs 中文），保存为 PDF
    if args.plot and all_model_data:
        # 指定的目标模型显示名称列表
        target_models = [
            'Claude Sonnet 4.5', 'DeepSeek-V3.1', 'Gemini 2.5 Flash', 'GPT-4o',
            'GPT-OSS-20B', 'Phi-4', 'Gemma-3-12B-IT', 'Llama-3.1-8B', 'Qwen-2.5-7B'
        ]
        # 按语言分组并过滤
        en_models = [(name, mids, avgs) for (lang, name, mids, avgs) in all_model_data 
                     if lang == 'en' and name in target_models]
        zh_models = [(name, mids, avgs) for (lang, name, mids, avgs) in all_model_data 
                     if lang == 'zh' and name in target_models]

        if en_models or zh_models:
            # 为每个目标模型分配固定的颜色和标记（统一左右）
            all_targets = target_models
            color_list = plt.cm.tab20.colors
            marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
            line_styles = ['-', '--', '-.', ':']
            model_style = {}
            for idx, model in enumerate(all_targets):
                model_style[model] = {
                    'color': color_list[idx % len(color_list)],
                    'marker': marker_list[idx % len(marker_list)],
                    'linestyle': line_styles[idx // len(marker_list) % len(line_styles)]
                }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 左图：英文模型
            if en_models:
                for name, mids, avgs in en_models:
                    style = model_style.get(name, {'color':'gray', 'marker':'o', 'linestyle':'-'})
                    ax1.plot(mids, avgs, marker=style['marker'], linestyle=style['linestyle'],
                             linewidth=1.5, markersize=5, color=style['color'], label=name)
                ax1.set_xlabel("Fused Text Length (characters)", fontsize=12, fontweight='bold')
                ax1.set_ylabel("Average Number of Problems", fontsize=12, fontweight='bold')
                ax1.set_title("EN", fontsize=14, fontweight='bold')
                ax1.grid(True, linestyle=':', alpha=0.5, color='gray')
                ax1.legend(loc='best', fontsize=8, frameon=True, edgecolor='black')
            else:
                ax1.text(0.5, 0.5, "No English models data", ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title("EN")

            # 右图：中文模型
            if zh_models:
                for name, mids, avgs in zh_models:
                    style = model_style.get(name, {'color':'gray', 'marker':'o', 'linestyle':'-'})
                    ax2.plot(mids, avgs, marker=style['marker'], linestyle=style['linestyle'],
                             linewidth=1.5, markersize=5, color=style['color'], label=name)
                ax2.set_xlabel("Fused Text Length (characters)", fontsize=12, fontweight='bold')
                ax2.set_ylabel("Average Number of Problems", fontsize=12, fontweight='bold')
                ax2.set_title("ZH", fontsize=14, fontweight='bold')
                ax2.grid(True, linestyle=':', alpha=0.5, color='gray')
                ax2.legend(loc='best', fontsize=8, frameon=True, edgecolor='black')
            else:
                ax2.text(0.5, 0.5, "No Chinese models data", ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("ZH")

            plt.suptitle("Average Total Problems vs. Fused Text Length", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            out_file = os.path.join(args.output_dir, "comparison_combined.pdf")
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"合并对比图已保存: {out_file}", file=sys.stderr)

if __name__ == "__main__":
    main()