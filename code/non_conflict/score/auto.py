#!/usr/bin/env python3
"""
新闻摘要评估自动评分汇总脚本
整合逻辑、语言、信息完整性三大维度的评分
排除 id 为 003 的数据，使用指数衰减公式计算得分
新增：S_fact = sqrt(S_info * S_logic) 及排名对比表格
"""

import json
import os
import math
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# ==================== 配置参数 ====================

# 基础路径配置。默认使用仓库内相对目录，开源用户可通过环境变量覆盖。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = os.getenv("DRIFT_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
OUTPUT_ROOT = os.getenv("DRIFT_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs", "non_conflict"))
EVAL_DIR = os.getenv("DRIFT_EVAL_DIR", os.path.join(OUTPUT_ROOT, "eval"))
SCORE_DIR = os.getenv("DRIFT_SCORE_DIR", os.path.join(OUTPUT_ROOT, "score"))

# 新闻总数（排除 003 后为 150 篇）
TOTAL_NEWS = 150

# 模型配置（已移除 gpt-4o-mini）
MODELS = {
    "qw7": {"path": "qw7", "display_name": "qwen-2.5-7b-instruct"},
    "gpt-4o": {"path": "gpt-4o", "display_name": "gpt-4o"},
    "deepseek-v3.1": {"path": "deepseek-v3.1", "display_name": "deepseek-v3.1"},
    "gemma": {"path": "gemma", "display_name": "gemma-3-12b-it"},
    "gpt-oss-20b": {"path": "gpt-oss-20b", "display_name": "gpt-oss-20b"},
    "llama": {"path": "llama", "display_name": "llama-3.1-8B-instruct"},
    "phi-4": {"path": "phi-4", "display_name": "phi-4"},
    "claude-sonnet-4-5": {"path": "claude-sonnet-4-5", "display_name": "claude-sonnet-4-5"},
    "gemini-2.5-flash": {"path": "gemini-2.5-flash", "display_name": "gemini-2.5-flash"}
}

# ==================== 修改后的 Lambda 参数 ====================
# 公式：Score = 100 * exp(-lambda * E_average)
# 新标准：
#   - 信息完整性：E_average=2 时得 50 分 -> λ = ln2 / 2 ≈ 0.346574
#   - 语言质量：   E_average=4 时得 50 分 -> λ = ln2 / 4 ≈ 0.173287
#   - 逻辑连贯性： E_average=1 时得 50 分 -> λ = ln2     ≈ 0.693147
LAMBDA_INFO = math.log(2) / 2
LAMBDA_LANG = math.log(2) / 4
LAMBDA_LOGIC = math.log(2)

# 问题小类中文名称映射
PROBLEM_SUBCATEGORY_ZH = {
    "time_order_inaccuracy": "时间顺序不准",
    "narrative_incoherence": "叙述不连贯",
    "causal_sequence_confusion": "因果混乱",
    "other_logical_issues": "其他逻辑问题",
    "stylistic_inconsistency": "风格不一致",
    "awkward_expression": "表达生硬",
    "redundant_repetition": "冗余重复",
    "inconsistent_terminology": "术语不一致",
    "tone_mismatch": "语气不匹配",
    "sentence_structure_problems": "句子结构问题",
    "other_language_issues": "其他语言问题",
    "global_issues": "全局问题",
    "paragraph_issues": "段落问题"
}

# 问题小类分类配置
LOGIC_SUBCATEGORIES = [
    "time_order_inaccuracy",
    "narrative_incoherence",
    "causal_sequence_confusion",
    "other_logical_issues"
]

LANG_SUBCATEGORIES = [
    "stylistic_inconsistency",
    "awkward_expression",
    "redundant_repetition",
    "inconsistent_terminology",
    "tone_mismatch",
    "sentence_structure_problems",
    "other_language_issues"
]

INFO_SUBCATEGORIES = [
    "global_issues",
    "paragraph_issues"
]


# ==================== 数据加载与过滤 ====================

def load_json_file(filepath: str) -> Any:
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件 {filepath} 时出错：{e}")
        return None


def filter_excluded_ids(news_list: List[Dict], exclude_ids: List[str] = ["003"]) -> List[Dict]:
    """排除指定 ID 的数据"""
    filtered = []
    excluded_count = 0
    for item in news_list:
        item_id = str(item.get("id", ""))
        if item_id in exclude_ids:
            excluded_count += 1
        else:
            filtered.append(item)
    if excluded_count > 0:
        print(f"   排除 {excluded_count} 条 ID 为 {exclude_ids} 的数据")
    return filtered


# ==================== 逻辑问题评分 ====================

def score_logic_single_news(news_item: dict) -> dict:
    """计算单篇新闻的逻辑问题数量（不涉及权重）"""
    if not news_item.get("evaluation_success", False):
        return {
            "score": 0.0,
            "problem_counts": {},
            "error": "Evaluation failed"
        }
    
    evaluation = news_item.get("evaluation", {})
    global_assessments = evaluation.get("global_evaluation", {}).get("global_assessments", [])
    
    total_problems = 0
    problem_counts = defaultdict(int)
    
    for assessment in global_assessments:
        if not assessment.get("problem_exists", False):
            continue
        evidence = assessment.get("evidence", {})
        categorized_problems = evidence.get("categorized_problems", {})
        
        for category, problems in categorized_problems.items():
            count = len(problems)
            if count == 0:
                continue
            total_problems += count
            problem_counts[category] += count
    
    # 简化得分：每个问题扣1分，最低0分（此得分仅为内部使用，最终评分表不使用）
    linear_score = max(0.0, 100.0 - total_problems)
    
    return {
        "score": round(linear_score, 2),
        "problem_counts": dict(problem_counts)
    }


def process_logic_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """处理逻辑评分文件"""
    data = load_json_file(input_path)
    if not data:
        return {}
    
    news_list = data.get("data", []) if isinstance(data, dict) else data
    news_list = filter_excluded_ids(news_list)
    
    scored_news = []
    total_score = 0.0
    total_items = 0
    overall_problem_counts = defaultdict(int)
    
    for news in news_list:
        result = score_logic_single_news(news)
        scored_item = {
            "original_id": news.get("id", None),
            "score": result["score"],
            "problem_counts": result["problem_counts"]
        }
        if "error" in result:
            scored_item["error"] = result["error"]
        scored_news.append(scored_item)
        
        if "error" not in result:
            total_score += result["score"]
            total_items += 1
            for cat, cnt in result["problem_counts"].items():
                overall_problem_counts[cat] += cnt
    
    original_average = round(total_score / total_items, 2) if total_items > 0 else 0.0
    total_problems = sum(overall_problem_counts.values())
    
    summary = {
        "total_news_scored": total_items,
        "original_average_score": original_average,
        "total_logical_problems": total_problems,
        "average_problems_per_news": round(total_problems / total_items, 2) if total_items > 0 else 0.0,
        "problem_type_breakdown": dict(overall_problem_counts)
    }
    
    output_data = {
        "summary": summary,
        "scores": scored_news
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_data


# ==================== 语言问题评分 ====================

def score_language_single_news(article_data: Dict[str, Any]) -> Tuple[float, int, Dict[str, int]]:
    """
    计算单篇新闻的语言问题数量（不涉及权重）
    """
    if not article_data.get("evaluation_success", False):
        return 0.0, 0, {}
    
    problem_classification = article_data.get("language_quality_evaluation", {}).get("problem_classification", {})
    detailed_categories = problem_classification.get("detailed_language_problem_categories", {})
    
    if not detailed_categories:
        return 100.0, 0, {}
    
    category_stats = {}
    if "categories" in detailed_categories and isinstance(detailed_categories["categories"], dict):
        category_stats = detailed_categories["categories"]
    elif isinstance(detailed_categories, dict) and detailed_categories:
        known_keys = set([
            "stylistic_inconsistency", "awkward_expression", "redundant_repetition",
            "inconsistent_terminology", "tone_mismatch", "sentence_structure_problems",
            "other_language_issues"
        ])
        if any(key in detailed_categories for key in known_keys):
            category_stats = detailed_categories
    
    total_problems = 0
    category_counts = {}
    
    for category_key, category_info in category_stats.items():
        if isinstance(category_info, (int, float)):
            count = int(category_info)
        elif isinstance(category_info, dict):
            count = category_info.get("count", 0)
            if not isinstance(count, (int, float)):
                count = 0
        else:
            continue
        
        if count > 0:
            total_problems += count
            category_counts[category_key] = count
    
    linear_score = max(0.0, 100.0 - total_problems)
    return linear_score, total_problems, category_counts


def process_lang_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """处理语言评分文件"""
    data = load_json_file(input_path)
    if not data:
        return {}
    
    articles_data = data.get("data", []) if isinstance(data, dict) else data
    articles_data = filter_excluded_ids(articles_data)
    
    scores_data = []
    total_problems_all = 0
    total_score_sum = 0.0
    valid_articles = 0
    category_summary = {cat: 0 for cat in [
        "stylistic_inconsistency", "awkward_expression", "redundant_repetition",
        "inconsistent_terminology", "tone_mismatch", "sentence_structure_problems",
        "other_language_issues"
    ]}
    
    for article in articles_data:
        score, problem_count, category_counts = score_language_single_news(article)
        
        article_id = article.get("id", f"article_{len(scores_data)}")
        scores_data.append({
            "id": article_id,
            "score": round(score, 2),
            "problem_count": problem_count,
            "category_counts": category_counts
        })
        
        if article.get("evaluation_success", False):
            total_problems_all += problem_count
            total_score_sum += score
            valid_articles += 1
            for category, count in category_counts.items():
                if category in category_summary:
                    category_summary[category] += count
    
    original_average_score = round(total_score_sum / valid_articles, 2) if valid_articles > 0 else 0.0
    
    category_distribution = {}
    for category, count in category_summary.items():
        if count > 0:
            chinese_name = PROBLEM_SUBCATEGORY_ZH.get(category, category)
            category_distribution[category] = {
                "english_name": category,
                "chinese_name": chinese_name,
                "count": count
            }
    
    output_data = {
        "total_articles": len(articles_data),
        "valid_articles": valid_articles,
        "original_average_score": original_average_score,
        "total_problems": total_problems_all,
        "average_problems_per_news": round(total_problems_all / valid_articles, 2) if valid_articles > 0 else 0.0,
        "scores": scores_data,
        "category_summary": category_distribution
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_data

# ==================== 信息完整性评分 ====================

def calculate_global_issues(global_assessments: List[Dict[str, Any]]) -> int:
    """计算全局信息完整性问题数量"""
    issue_count = 0
    for assessment in global_assessments:
        evidence = assessment.get("evidence", {})
        unused_list = evidence.get("unused_content_analysis", [])
        for item in unused_list:
            impact = item.get("omission_impact", "No")
            if impact in ("Yes", "Partial"):
                issue_count += 1
    return issue_count


def calculate_paragraph_issues(paragraph_assessments: List[Dict[str, Any]]) -> int:
    """计算段落信息完整性问题数量"""
    issue_count = 0
    for assessment in paragraph_assessments:
        evidence = assessment.get("evidence", {})
        comparisons = evidence.get("evidence_comparison", [])
        for comp in comparisons:
            inclusion = comp.get("included_in_fused", "No")
            if inclusion == "No":
                issue_count += 1
    return issue_count


def process_info_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """处理信息完整性评分文件"""
    data = load_json_file(input_path)
    if not data:
        return {}
    
    news_list = data.get("data", []) if isinstance(data, dict) else data
    news_list = filter_excluded_ids(news_list)
    
    results = []
    total_issues_sum = 0
    total_global_issues_sum = 0
    total_paragraph_issues_sum = 0
    
    for item in news_list:
        dual_eval = item.get("dual_evaluation", {})
        global_assessments = dual_eval.get("global_evaluation", {}).get("global_assessments", [])
        paragraph_assessments = dual_eval.get("paragraph_evaluation", {}).get("paragraph_assessments", [])
        
        global_issues = calculate_global_issues(global_assessments)
        para_issues = calculate_paragraph_issues(paragraph_assessments)
        
        total_issues = global_issues + para_issues
        
        results.append({
            "global_issues": global_issues,
            "paragraph_issues": para_issues,
            "total_issues": total_issues
        })
        
        total_issues_sum += total_issues
        total_global_issues_sum += global_issues
        total_paragraph_issues_sum += para_issues
    
    output_data = {
        "summary": {
            "total_news_count": len(results),
            "total_global_issues": total_global_issues_sum,
            "total_paragraph_issues": total_paragraph_issues_sum,
            "total_issues_across_all_news": total_issues_sum,
            "average_problems_per_news": round(total_issues_sum / len(results), 2) if results else 0.0
        },
        "per_news_scores": results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_data


# ==================== 得分计算 ====================

def calculate_category_score(total_problems: int, lambda_val: float, total_news: int) -> float:
    """计算类别得分：100 * exp(-lambda * E_average)"""
    if total_problems == 0:
        return 100.0
    
    E_average = total_problems / total_news
    score = 100 * math.exp(-lambda_val * E_average)
    return round(score, 2)


def calculate_total_score(logic_score: float, lang_score: float, info_score: float) -> float:
    """计算总分数：几何平均数"""
    if logic_score <= 0 or lang_score <= 0 or info_score <= 0:
        return 0.0
    
    product = logic_score * lang_score * info_score
    if product <= 0:
        return 0.0
    
    return round(math.pow(product, 1/3), 2)


def calculate_fact_score(info_score: float, logic_score: float) -> float:
    """计算事实性得分：S_fact = sqrt(S_info * S_logic)"""
    if info_score <= 0 or logic_score <= 0:
        return 0.0
    return round(math.sqrt(info_score * logic_score), 2)


# ==================== 文件查找 ====================

def find_score_files(base_dir: str, model_dir: str, language: str, score_type: str) -> List[str]:
    """查找评分文件"""
    model_path = os.path.join(base_dir, model_dir)
    
    if not os.path.exists(model_path):
        return []
    
    found_files = []
    try:
        for filename in os.listdir(model_path):
            file_lower = filename.lower()
            if (language in file_lower and 
                (model_dir in file_lower or model_path.split('/')[-1] in file_lower) and 
                (score_type in file_lower or 
                 (score_type == "lang" and "language" in file_lower))):
                found_files.append(os.path.join(model_path, filename))
    except FileNotFoundError:
        pass
    
    return found_files


# ==================== 报告生成 ====================

def generate_scoring_table(all_results: Dict, total_news: int) -> str:
    """生成评分结果汇总表（增加 S_fact 列）"""
    table = "\n### 评分结果汇总表\n\n"
    table += "| 模型 | 语言 | S_logic | S_lang | S_info | S_fact | S_total |\n"
    table += "|------|------|---------|--------|--------|--------|---------|\n"
    
    sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
    
    for model in sorted_models:
        display_name = MODELS.get(model, {}).get("display_name", model)
        
        for language in ["zh", "en"]:
            if language in all_results[model]:
                lang_name = "中文" if language == "zh" else "英文"
                data = all_results[model][language]
                if all(key in data for key in ["S_logic", "S_lang", "S_info", "S_total", "S_fact"]):
                    table += f"| {display_name} | {lang_name} | {data['S_logic']:.2f} | {data['S_lang']:.2f} | {data['S_info']:.2f} | {data['S_fact']:.2f} | {data['S_total']:.2f} |\n"
    
    table += "\n**评分公式说明:**\n\n"
    table += "1. 各类别得分计算公式：$S = 100 \\cdot e^{-\\lambda \\cdot E_{\\text{average}}}$\n"
    table += f"2. 其中 $E_{{\\text{{average}}}} = \\frac{{\\text{{总问题数}}}}{{{total_news}}}$\n"
    table += "3. 各参数设置:\n"
    table += f"   - 信息完整性 ($\\lambda_1 = {LAMBDA_INFO:.4f}$): $E_{{average}}=2$ 时对应 50 分\n"
    table += f"   - 语言质量 ($\\lambda_2 = {LAMBDA_LANG:.4f}$): $E_{{average}}=4$ 时对应 50 分\n"
    table += f"   - 逻辑连贯性 ($\\lambda_3 = {LAMBDA_LOGIC:.4f}$): $E_{{average}}=1$ 时对应 50 分\n"
    table += "4. 总得分计算公式：$S_{\\text{total}} = \\sqrt[3]{S_{\\text{logic}} \\times S_{\\text{lang}} \\times S_{\\text{info}}}$\n"
    table += "5. **新增**：事实性得分 $S_{\\text{fact}} = \\sqrt{S_{\\text{info}} \\times S_{\\text{logic}}}$\n"
    table += f"6. **已排除 ID 为 003 的数据，新闻总数：{total_news} 篇**\n"
    
    return table


def generate_fact_vs_total_table(all_results: Dict) -> str:
    """生成 S_fact 与 S_total 的排名对比表格（分中英文）"""
    table = "\n### S_fact 与 S_total 排名对比\n\n"
    table += "本表展示各模型在中英文上 S_fact 与 S_total 的得分及排名，以分析两者之间的关系。\n\n"
    
    for language in ["zh", "en"]:
        lang_name = "中文" if language == "zh" else "英文"
        table += f"#### {lang_name}\n\n"
        
        # 收集该语言下所有模型的数据
        rows = []
        for model in all_results:
            if language in all_results[model]:
                data = all_results[model][language]
                if "S_fact" in data and "S_total" in data:
                    rows.append({
                        "model": model,
                        "display_name": MODELS.get(model, {}).get("display_name", model),
                        "S_fact": data["S_fact"],
                        "S_total": data["S_total"]
                    })
        
        if not rows:
            table += "无数据\n\n"
            continue
        
        # 按 S_fact 降序排名（排名1为最高分）
        sorted_by_fact = sorted(rows, key=lambda x: x["S_fact"], reverse=True)
        for idx, row in enumerate(sorted_by_fact, 1):
            row["fact_rank"] = idx
        
        # 按 S_total 降序排名
        sorted_by_total = sorted(rows, key=lambda x: x["S_total"], reverse=True)
        for idx, row in enumerate(sorted_by_total, 1):
            row["total_rank"] = idx
        
        # 重新按模型显示名称排序以便输出
        sorted_by_model = sorted(rows, key=lambda x: x["display_name"])
        
        table += "| 模型 | S_fact | S_fact排名 | S_total | S_total排名 | 排名差 (Total - Fact) |\n"
        table += "|------|--------|------------|---------|-------------|------------------------|\n"
        for row in sorted_by_model:
            diff = row["total_rank"] - row["fact_rank"]
            table += f"| {row['display_name']} | {row['S_fact']:.2f} | {row['fact_rank']} | {row['S_total']:.2f} | {row['total_rank']} | {diff:+d} |\n"
        table += "\n"
    
    table += "**排名说明**：排名数值越小表示得分越高。排名差为正表示 S_total 排名低于 S_fact（即 S_total 相对较差），为负则表示 S_total 排名优于 S_fact。\n"
    return table


def generate_problem_summary_table(all_results: Dict) -> str:
    """生成问题数量汇总表"""
    table = "\n### 问题数量汇总表\n\n"
    table += "| 模型 | 语言 | 逻辑问题数 | 语言问题数 | 信息问题数 | 总问题数 |\n"
    table += "|------|------|-----------|-----------|-----------|---------|\n"
    
    sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
    
    for model in sorted_models:
        display_name = MODELS.get(model, {}).get("display_name", model)
        
        for language in ["zh", "en"]:
            if language in all_results[model]:
                lang_name = "中文" if language == "zh" else "英文"
                data = all_results[model][language]
                logic = data.get("total_logic_problems", 0)
                lang = data.get("total_lang_problems", 0)
                info = data.get("total_info_problems", 0)
                total = logic + lang + info
                table += f"| {display_name} | {lang_name} | {logic} | {lang} | {info} | {total} |\n"
    
    return table


def generate_logic_subcategory_table(all_results: Dict) -> str:
    """生成逻辑问题小类统计表"""
    table = "\n### 逻辑问题小类详细统计\n\n"
    
    for language in ["zh", "en"]:
        lang_name = "中文" if language == "zh" else "英文"
        table += f"#### {lang_name}逻辑问题小类统计\n\n"
        table += "| 模型 | 时间顺序不准 | 叙述不连贯 | 因果混乱 | 其他逻辑问题 | 逻辑问题总计 |\n"
        table += "|------|-------------|------------|----------|--------------|-------------|\n"
        
        sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
        
        for model in sorted_models:
            if language in all_results[model]:
                display_name = MODELS.get(model, {}).get("display_name", model)
                data = all_results[model][language]
                logic_problems = data.get("logic_problems", {})
                
                time_order = logic_problems.get("time_order_inaccuracy", 0)
                narrative = logic_problems.get("narrative_incoherence", 0)
                causal = logic_problems.get("causal_sequence_confusion", 0)
                other = logic_problems.get("other_logical_issues", 0)
                total = data.get("total_logic_problems", 0)
                
                table += f"| {display_name} | {time_order} | {narrative} | {causal} | {other} | {total} |\n"
        
        table += "\n"
    
    return table


def generate_lang_subcategory_table(all_results: Dict) -> str:
    """生成语言问题小类统计表"""
    table = "\n### 语言问题小类详细统计\n\n"
    
    for language in ["zh", "en"]:
        lang_name = "中文" if language == "zh" else "英文"
        table += f"#### {lang_name}语言问题小类统计\n\n"
        table += "| 模型 | 风格不一致 | 表达生硬 | 冗余重复 | 术语不一致 | 语气不匹配 | 句子结构问题 | 其他语言问题 | 语言问题总计 |\n"
        table += "|------|-----------|----------|----------|------------|------------|--------------|--------------|-------------|\n"
        
        sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
        
        for model in sorted_models:
            if language in all_results[model]:
                display_name = MODELS.get(model, {}).get("display_name", model)
                data = all_results[model][language]
                lang_problems = data.get("lang_problems", {})
                
                stylistic = lang_problems.get("stylistic_inconsistency", 0)
                awkward = lang_problems.get("awkward_expression", 0)
                redundant = lang_problems.get("redundant_repetition", 0)
                terminology = lang_problems.get("inconsistent_terminology", 0)
                tone = lang_problems.get("tone_mismatch", 0)
                sentence = lang_problems.get("sentence_structure_problems", 0)
                other = lang_problems.get("other_language_issues", 0)
                total = data.get("total_lang_problems", 0)
                
                table += f"| {display_name} | {stylistic} | {awkward} | {redundant} | {terminology} | {tone} | {sentence} | {other} | {total} |\n"
        
        table += "\n"
    
    return table


def generate_info_subcategory_table(all_results: Dict) -> str:
    """生成信息完整性问题小类统计表"""
    table = "\n### 信息完整性问题小类详细统计\n\n"
    
    for language in ["zh", "en"]:
        lang_name = "中文" if language == "zh" else "英文"
        table += f"#### {lang_name}信息完整性问题小类统计\n\n"
        table += "| 模型 | 全局问题 | 段落问题 | 信息问题总计 |\n"
        table += "|------|----------|----------|-------------|\n"
        
        sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
        
        for model in sorted_models:
            if language in all_results[model]:
                display_name = MODELS.get(model, {}).get("display_name", model)
                data = all_results[model][language]
                info_problems = data.get("info_problems", {})
                
                global_issues = info_problems.get("global_issues", 0)
                paragraph_issues = info_problems.get("paragraph_issues", 0)
                total = data.get("total_info_problems", 0)
                
                table += f"| {display_name} | {global_issues} | {paragraph_issues} | {total} |\n"
        
        table += "\n"
    
    return table


def generate_all_subcategories_combined_table(all_results: Dict) -> str:
    """生成所有问题小类合并统计表"""
    table = "\n### 所有问题小类合并统计\n\n"
    
    for language in ["zh", "en"]:
        lang_name = "中文" if language == "zh" else "英文"
        table += f"#### {lang_name}所有问题小类合并统计\n\n"
        table += "| 模型 | "
        
        # 逻辑小类
        for subcat in LOGIC_SUBCATEGORIES:
            table += f"{PROBLEM_SUBCATEGORY_ZH.get(subcat, subcat)} | "
        # 语言小类
        for subcat in LANG_SUBCATEGORIES:
            table += f"{PROBLEM_SUBCATEGORY_ZH.get(subcat, subcat)} | "
        # 信息小类
        for subcat in INFO_SUBCATEGORIES:
            table += f"{PROBLEM_SUBCATEGORY_ZH.get(subcat, subcat)} | "
        
        table += "逻辑总计 | 语言总计 | 信息总计 | 总问题数 |\n"
        
        # 分隔线
        table += "|------|"
        for _ in LOGIC_SUBCATEGORIES + LANG_SUBCATEGORIES + INFO_SUBCATEGORIES + ["逻辑总计", "语言总计", "信息总计", "总问题数"]:
            table += "---|"
        table += "\n"
        
        sorted_models = sorted(all_results.keys(), key=lambda x: MODELS.get(x, {}).get("display_name", x))
        
        for model in sorted_models:
            if language in all_results[model]:
                display_name = MODELS.get(model, {}).get("display_name", model)
                data = all_results[model][language]
                
                table += f"| {display_name} | "
                
                # 逻辑小类
                logic_problems = data.get("logic_problems", {})
                for subcat in LOGIC_SUBCATEGORIES:
                    table += f"{logic_problems.get(subcat, 0)} | "
                
                # 语言小类
                lang_problems = data.get("lang_problems", {})
                for subcat in LANG_SUBCATEGORIES:
                    table += f"{lang_problems.get(subcat, 0)} | "
                
                # 信息小类
                info_problems = data.get("info_problems", {})
                for subcat in INFO_SUBCATEGORIES:
                    table += f"{info_problems.get(subcat, 0)} | "
                
                # 总计
                logic_total = data.get("total_logic_problems", 0)
                lang_total = data.get("total_lang_problems", 0)
                info_total = data.get("total_info_problems", 0)
                total_all = logic_total + lang_total + info_total
                
                table += f"{logic_total} | {lang_total} | {info_total} | {total_all} |\n"
        
        table += "\n"
    
    return table


def save_report(all_results: Dict, output_file: str, total_news: int):
    """保存汇总报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 多模型评估结果详细汇总报告\n\n")
        f.write(f"**新闻总数：{total_news} 篇（已排除 ID 为 003 的数据）**\n\n")
        
        f.write("## 问题数量汇总\n")
        f.write(generate_problem_summary_table(all_results))
        
        f.write("## 评分结果\n")
        f.write(generate_scoring_table(all_results, total_news))
        
        f.write("## S_fact 与 S_total 排名对比\n")
        f.write(generate_fact_vs_total_table(all_results))
        
        f.write("## 问题小类详细统计\n")
        f.write(generate_logic_subcategory_table(all_results))
        f.write(generate_lang_subcategory_table(all_results))
        f.write(generate_info_subcategory_table(all_results))
        
        f.write("## 所有问题小类合并统计\n")
        f.write(generate_all_subcategories_combined_table(all_results))
        
        f.write("\n## Lambda 参数详情\n\n")
        f.write("各参数计算过程（新标准）:\n\n")
        f.write("1. **信息完整性 (info)**\n")
        f.write(f"   - 条件：E_average = 2 时，得分 = 50\n")
        f.write(f"   - 方程：50 = 100 * exp(-λ1 * 2)\n")
        f.write(f"   - 解得：λ1 = -ln(0.5)/2 = ln(2)/2 ≈ {LAMBDA_INFO:.6f}\n\n")
        
        f.write("2. **语言质量 (lang)**\n")
        f.write(f"   - 条件：E_average = 4 时，得分 = 50\n")
        f.write(f"   - 方程：50 = 100 * exp(-λ2 * 4)\n")
        f.write(f"   - 解得：λ2 = -ln(0.5)/4 = ln(2)/4 ≈ {LAMBDA_LANG:.6f}\n\n")
        
        f.write("3. **逻辑连贯性 (logic)**\n")
        f.write(f"   - 条件：E_average = 1 时，得分 = 50\n")
        f.write(f"   - 方程：50 = 100 * exp(-λ3 * 1)\n")
        f.write(f"   - 解得：λ3 = -ln(0.5) = ln(2) ≈ {LAMBDA_LOGIC:.6f}\n\n")


# ==================== 主流程 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻摘要评估自动评分汇总脚本")
    parser.add_argument("--eval-dir", type=str, default=EVAL_DIR, help="评估结果目录")
    parser.add_argument("--score-dir", type=str, default=SCORE_DIR, help="评分结果输出目录")
    parser.add_argument("--total-news", type=int, default=TOTAL_NEWS, help="新闻总数")
    args = parser.parse_args()
    
    total_news = args.total_news
    
    os.makedirs(args.score_dir, exist_ok=True)
    
    print("=" * 80)
    print("新闻摘要评估自动评分汇总脚本")
    print("=" * 80)
    print(f"评估目录：{args.eval_dir}")
    print(f"输出目录：{args.score_dir}")
    print(f"新闻总数：{total_news}（已排除 ID 为 003 的数据）")
    print("=" * 80)
    
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for model_key, model_config in MODELS.items():
        model_path = model_config["path"]
        print(f"\n📊 处理模型：{model_key} ({model_config['display_name']})")
        
        for language in ["zh", "en"]:
            print(f"  处理语言：{language}")
            
            # 逻辑评分文件
            logic_files = find_score_files(args.eval_dir, model_path, language, "logic")
            if logic_files:
                output_path = os.path.join(args.score_dir, model_key, f"{language}_{model_key}_logic.json")
                logic_data = process_logic_file(logic_files[0], output_path)
                if logic_data and "summary" in logic_data:
                    summary = logic_data["summary"]
                    all_results[model_key][language]["logic_problems"] = summary.get("problem_type_breakdown", {})
                    all_results[model_key][language]["total_logic_problems"] = summary.get("total_logical_problems", 0)
                    all_results[model_key][language]["S_logic"] = calculate_category_score(
                        summary.get("total_logical_problems", 0), LAMBDA_LOGIC, total_news
                    )
                    print(f"    ✅ 逻辑评分完成，问题数：{summary.get('total_logical_problems', 0)}")
            
            # 语言评分文件
            lang_files = find_score_files(args.eval_dir, model_path, language, "lang")
            if lang_files:
                output_path = os.path.join(args.score_dir, model_key, f"{language}_{model_key}_lang.json")
                lang_data = process_lang_file(lang_files[0], output_path)
                if lang_data:
                    category_summary = lang_data.get("category_summary", {})
                    lang_problems = {k: v.get("count", 0) for k, v in category_summary.items()}
                    all_results[model_key][language]["lang_problems"] = lang_problems
                    all_results[model_key][language]["total_lang_problems"] = lang_data.get("total_problems", 0)
                    all_results[model_key][language]["S_lang"] = calculate_category_score(
                        lang_data.get("total_problems", 0), LAMBDA_LANG, total_news
                    )
                    print(f"    ✅ 语言评分完成，问题数：{lang_data.get('total_problems', 0)}")
            
            # 信息完整性评分文件
            info_files = find_score_files(args.eval_dir, model_path, language, "info")
            if info_files:
                output_path = os.path.join(args.score_dir, model_key, f"{language}_{model_key}_info.json")
                info_data = process_info_file(info_files[0], output_path)
                if info_data and "summary" in info_data:
                    summary = info_data["summary"]
                    all_results[model_key][language]["info_problems"] = {
                        "global_issues": summary.get("total_global_issues", 0),
                        "paragraph_issues": summary.get("total_paragraph_issues", 0)
                    }
                    all_results[model_key][language]["total_info_problems"] = summary.get("total_issues_across_all_news", 0)
                    all_results[model_key][language]["S_info"] = calculate_category_score(
                        summary.get("total_issues_across_all_news", 0), LAMBDA_INFO, total_news
                    )
                    print(f"    ✅ 信息评分完成，问题数：{summary.get('total_issues_across_all_news', 0)}")
            
            # 计算总分和 S_fact
            if all(key in all_results[model_key][language] for key in ["S_logic", "S_lang", "S_info"]):
                S_logic = all_results[model_key][language]["S_logic"]
                S_lang = all_results[model_key][language]["S_lang"]
                S_info = all_results[model_key][language]["S_info"]
                
                # 计算 S_total
                all_results[model_key][language]["S_total"] = calculate_total_score(S_logic, S_lang, S_info)
                # 计算 S_fact
                all_results[model_key][language]["S_fact"] = calculate_fact_score(S_info, S_logic)
                
                data = all_results[model_key][language]
                print(f"    📈 得分：逻辑={data['S_logic']:.2f}, 语言={data['S_lang']:.2f}, "
                      f"信息={data['S_info']:.2f}, 事实={data['S_fact']:.2f}, 总分={data['S_total']:.2f}")
    
    print("\n" + "=" * 80)
    print("生成汇总报告...")
    print("=" * 80)
    
    report_file = os.path.join(args.score_dir, "detailed_summary_report.md")
    save_report(all_results, report_file, total_news)
    print(f"✅ 报告已保存到：{report_file}")
    
    print("\n" + "=" * 80)
    print("评分结果汇总表:")
    print("=" * 80)
    print(generate_scoring_table(all_results, total_news))
    
    print("\n" + "=" * 80)
    print("S_fact 与 S_total 排名对比:")
    print("=" * 80)
    print(generate_fact_vs_total_table(all_results))
    
    print("\n" + "=" * 80)
    print("🎉 所有评分完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
