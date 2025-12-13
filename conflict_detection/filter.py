import json
import os
import sys
from typing import List, Dict, Any, Set, Tuple

# 导入语言检测库
try:
    import langid
except ImportError:
    print("❌ 错误：缺少 'langid' 库。请运行：pip install langid")
    sys.exit(1)

# --- 配置 ---
INPUT_FILE_PATH = "/data1/rjj/a.test/conflict_detection/evaluation_results_eval_doubao-1-5-pro-256k-250115_recalculated_qwen_7b.json"
# 过滤并重新计算后的输出文件将添加此前缀
OUTPUT_FILE_SUFFIX = "_recalculated_segregated.json"

# 过滤条件二：期望的 _processing_log 内容（精确匹配）
REQUIRED_LOG_ITEMS: Set[str] = {
    "json_loads_ok_direct",
    "parsed_conflict_json_via_markers"
}
REQUIRED_LOG_LENGTH: int = len(REQUIRED_LOG_ITEMS)

def calculate_metrics(TP: int, FN: int, FP: int) -> Dict[str, float]:
    """计算 Recall, Precision 和 F1 Score，并返回所有计数。"""
    
    total_relevant = TP + FN
    Recall = TP / total_relevant if total_relevant > 0 else 0.0
    
    total_retrieved = TP + FP
    Precision = TP / total_retrieved if total_retrieved > 0 else 0.0
        
    precision_plus_recall = Precision + Recall
    F1 = 2 * (Precision * Recall) / precision_plus_recall if precision_plus_recall > 0 else 0.0
        
    return {
        "total_injected_conflicts": total_relevant,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "Recall": round(Recall, 4),
        "Precision": round(Precision, 4),
        "F1": round(F1, 4)
    }

def aggregate_stats(data_list: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """累加给定数据列表中的 TP, FN, FP。"""
    TP, FN, FP = 0, 0, 0
    
    for item in data_list:
        audit = item.get("conflict_detection_evaluation", {}).get("audit_summary", {})
        
        # 安全地累加各项统计值
        TP += audit.get("TP", 0)
        FN += audit.get("FN", 0)
        FP += audit.get("FP", 0)
        
    return TP, FN, FP

def filter_and_recalculate_segregated(input_path: str, output_suffix: str) -> None:
    """
    加载评估结果 JSON 文件，执行过滤，分语言计算，并将所有结果合并存储。
    """
    print(f"🔄 正在加载文件: {input_path}")
    
    # 1. 加载文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data_structure: Dict[str, Any] = json.load(f)
            
    except FileNotFoundError:
        print(f"❌ 错误：输入文件不存在于路径: {input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：解析 JSON 文件失败: {e}")
        sys.exit(1)
        
    # 提取核心数据列表
    original_data: List[Dict[str, Any]] = data_structure.get("data", [])
    original_count = len(original_data)
    
    print(f"✅ 文件加载成功。原始记录数: {original_count}")

    # 2. 过滤逻辑 (生成 filtered_data)
    filtered_data: List[Dict[str, Any]] = []

    for item in original_data:
        # 过滤条件一：删除 'note': 'no_shared_entities' 的记录
        note = item.get("injection_result", {}).get("summary", {}).get("note")
        if note == "no_shared_entities":
            continue

        # 过滤条件二：删除 _processing_log 内容不完全等于期望值的记录
        processing_log: List[str] = item.get("_processing_log", [])
        
        # 检查长度和内容是否精确匹配
        if len(processing_log) != REQUIRED_LOG_LENGTH or set(processing_log) != REQUIRED_LOG_ITEMS:
            continue
            
        filtered_data.append(item)

    # 3. 分语言分类
    en_data: List[Dict[str, Any]] = []
    zh_data: List[Dict[str, Any]] = []
    other_data: List[Dict[str, Any]] = []
    
    for item in filtered_data:
        input_text = item.get("input_text", "")
        if not input_text:
            continue
            
        try:
            lang, _ = langid.classify(input_text)
            if lang == 'zh':
                zh_data.append(item)
            elif lang == 'en':
                en_data.append(item)
            else:
                other_data.append(item)
        except Exception:
            other_data.append(item) # 无法分类的视为其他

    # 4. 独立统计计算
    
    # 4.1. 全部过滤数据统计 (All Filtered Data)
    TP_all, FN_all, FP_all = aggregate_stats(filtered_data)
    metrics_all = calculate_metrics(TP_all, FN_all, FP_all)
    
    # 4.2. 英文数据统计 (English Data)
    TP_en, FN_en, FP_en = aggregate_stats(en_data)
    metrics_en = calculate_metrics(TP_en, FN_en, FP_en)

    # 4.3. 中文数据统计 (Chinese Data)
    TP_zh, FN_zh, FP_zh = aggregate_stats(zh_data)
    metrics_zh = calculate_metrics(TP_zh, FN_zh, FP_zh)

    # 5. 更新 JSON 结构
    
    new_count = len(filtered_data)
    data_structure["data"] = filtered_data # 保留过滤后的全部数据
    data_structure["total_items"] = new_count
    
    # 确保 evaluation_stats 路径存在
    if "evaluation_stats" not in data_structure:
        data_structure["evaluation_stats"] = {}
        
    stats = data_structure["evaluation_stats"]
    
    # 更新全部数据统计 (与原有逻辑一致)
    stats["conflict_detection"] = metrics_all
    
    # 新增分语言统计
    stats["conflict_detection_en"] = metrics_en
    stats["conflict_detection_zh"] = metrics_zh
    
    # 6. 确定输出路径并保存
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, name + output_suffix)
    
    print(f"✨ 过滤和统计完成。保留记录数: {new_count} (已删除 {original_count - new_count} 条)")
    print(f"📦 分语言统计结果：英文记录数={len(en_data)}, 中文记录数={len(zh_data)}, 其他记录数={len(other_data)}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print(f"📊 重新计算的指标 (全部过滤数据, N={new_count}):")
    print(f"   Recall: {metrics_all['Recall']}, Precision: {metrics_all['Precision']}, F1: {metrics_all['F1']}")
    print("-" * 80)
    print(f"🌍 英文指标 (N={len(en_data)}):")
    print(f"   Recall: {metrics_en['Recall']}, Precision: {metrics_en['Precision']}, F1: {metrics_en['F1']}")
    print("-" * 80)
    print(f"🇨🇳 中文指标 (N={len(zh_data)}):")
    print(f"   Recall: {metrics_zh['Recall']}, Precision: {metrics_zh['Precision']}, F1: {metrics_zh['F1']}")
    print("="*80)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_structure, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已合并保存到: {output_path}")
    except Exception as e:
        print(f"❌ 错误：保存文件失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    filter_and_recalculate_segregated(INPUT_FILE_PATH, OUTPUT_FILE_SUFFIX)