#!/usr/bin/env python3
# coding: utf-8
"""
Evaluator Only: Use LLM as independent auditor to evaluate
how well the FUSION model detected injected conflicts.
Supports: Breakpoint continuation, Extended context (1000 chars), and CSI/Diagnostic aggregation.
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
from collections import defaultdict

# Make the repository's canonical shared configuration importable when this
# script is run directly from its track subdirectory.
COMMON_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
if COMMON_CODE_DIR not in sys.path:
    sys.path.insert(0, COMMON_CODE_DIR)

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Try to import config; fallback if not available
try:
    from config import (
        get_model_deployment_config,
        DEFAULT_TEACHER_MODEL,
        MAX_WORKERS,
        REQUEST_TIMEOUT,
        LOG_LEVEL,
        DEBUG_MODE,
        ENABLE_PROGRESS_BAR,
        API_RETRY_LIMIT
    )
except ImportError:
    # Fallback config
    DEFAULT_TEACHER_MODEL = "gpt-4o"
    MAX_WORKERS = 4
    REQUEST_TIMEOUT = 120.0
    LOG_LEVEL = "INFO"
    DEBUG_MODE = False
    ENABLE_PROGRESS_BAR = True
    API_RETRY_LIMIT = 3

    def get_model_deployment_config(model_name: str):
        return {
            "model": "doubao-seed-1-8-251228",
            "temperature": 0.0,
            "max_tokens": 4096,
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        }

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# === 断点续传管理类 ===
class CheckpointManager:
    def __init__(self, output_path: str):
        self.jsonl_path = output_path.replace('.json', '.jsonl')
        self.lock = Lock()
        self.processed_ids = set()

        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 使用 input_text 或 fused 的前50个字符作为指纹
                        fingerprint = self._get_fingerprint(data)
                        self.processed_ids.add(fingerprint)
                    except:
                        continue
            logger.info(f"Detected checkpoint: skipping {len(self.processed_ids)} items.")

    def _get_fingerprint(self, item: Dict) -> str:
        text = item.get("input_text", "") or extract_fused_content_from_item(item)
        return str(hash(text[:200]))

    def is_processed(self, item: Dict) -> bool:
        return self._get_fingerprint(item) in self.processed_ids

    def write(self, item: Dict):
        with self.lock:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

class EvaluationStats:
    def __init__(self):
        self.model_name = ""
        self.call_count = 0
        self.total_injected = 0

        # Phase 1 指标细化
        self.tp_reported = 0
        self.tp_silent = 0
        self.missed = 0       # FN
        self.false_positives = 0  # FP

        # Phase 2 诊断指标对齐 (MS, LN, RH, PF)
        self.n_majority_success = 0         # MS
        self.n_logical_neutral = 0          # LN (替代了原来的 selection_success)
        self.n_reconciliation_hallucination = 0 # RH
        self.n_propagation_failure = 0      # PF

        self.total_recall_score = 0.0
        self.diag_calls = 0

        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.call_count += 1

    def update(self, injected: int, tp_rep: int, tp_sil: int, missed: int, fp: int):
        with self.lock:
            self.total_injected += injected
            self.tp_reported += tp_rep
            self.tp_silent += tp_sil
            self.missed += missed
            self.false_positives += fp

    def update_diagnostic(self, ms: int, ln: int, rh: int, pf: int, recall: float):
        with self.lock:
            self.n_majority_success += ms
            self.n_logical_neutral += ln
            self.n_reconciliation_hallucination += rh
            self.n_propagation_failure += pf
            self.total_recall_score += recall
            self.diag_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_tp = self.tp_reported + self.tp_silent

            # 基础指标计算
            recall = total_tp / self.total_injected if self.total_injected > 0 else 0.0
            precision = self.tp_reported / (self.tp_reported + self.false_positives) if (self.tp_reported + self.false_positives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # 高阶诊断计算
            avg_recall = self.total_recall_score / self.diag_calls if self.diag_calls > 0 else 0.0
            silent_rate = self.tp_silent / total_tp if total_tp > 0 else 0.0

            return {
                "evaluator_model": self.model_name,
                "total_llm_calls": self.call_count,
                "phase1_awareness_global": {
                    "total_injected_conflicts": self.total_injected,
                    "TP_total": total_tp,
                    "TP_reported": self.tp_reported,
                    "TP_silent": self.tp_silent,
                    "FN": self.missed,
                    "FP": self.false_positives,
                    "global_recall": round(recall, 4),
                    "global_precision": round(precision, 4),
                    "global_f1": round(f1, 4),
                    "silent_processing_rate": round(silent_rate, 4) # 核心关注点1
                },
                "phase2_resolution_global": {
                    "MS_majority_success": self.n_majority_success,
                    "LN_logical_neutral": self.n_logical_neutral, # 核心关注点2
                    "RH_reconciliation_hallucination": self.n_reconciliation_hallucination,
                    "PF_propagation_failure": self.n_propagation_failure
                },
                "diagnostic_summary": {
                    "avg_item_recall": round(avg_recall, 4)
                }
            }
eval_stats = EvaluationStats()

# (以下 init_llm_from_config, safe_llm_invoke, extract_fused_content_from_item 函数逻辑保持不变)
def init_llm_from_config(model_name: str):
    cfg = get_model_deployment_config(model_name)
    model_cfg = cfg.get("model_name", cfg.get("model", model_name))
    api_key = cfg.get("api_key", "")
    if api_key == "EMPTY": api_key = "vllm"
    eval_stats.model_name = model_cfg
    return ChatOpenAI(model_name=model_cfg, temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 4096),
                      openai_api_key=api_key, openai_api_base=cfg.get("base_url", "https://api.openai.com/v1"),
                      request_timeout=REQUEST_TIMEOUT, max_retries=API_RETRY_LIMIT)

def safe_llm_invoke(llm, messages, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            eval_stats.increment()
            return llm.invoke(messages)
        except Exception as e:
            if attempt < max_retries: time.sleep(base_delay * (2 ** attempt))
            else: return None

def extract_fused_content_from_item(item):
    fused = item.get("fused_content", "") or item.get("fused", "")
    if not fused.strip() and "data" in item and isinstance(item["data"], list):
        for d in item["data"]:
            inner = d.get("fused_content", "") or d.get("fused", "")
            if inner.strip(): return inner
    return fused
def build_conflict_audit_prompt(injected_conflicts, fusion_conflicts, fused_text, versions):
    # 1. 准备数据
    injected_lines = [
        f"{i+1}. 实体: '{c.get('entity_text', c.get('entity', '未知'))}', {c.get('version_id', '未知')} 将 '{c.get('original_text', '')}' 改为 '{c.get('modified_text', '')}'"
        for i, c in enumerate(injected_conflicts)
    ]
    injected_block_str = "\n".join(injected_lines) if injected_lines else "无注入冲突"

    fusion_lines = [
        f"{i+1}. 实体: '{fc.get('entity', '未知')}', 决定: '{fc.get('final_decision', '')}'"
        for i, fc in enumerate(fusion_conflicts)
    ]
    fusion_block_str = "\n".join(fusion_lines) if fusion_lines else "融合模型未报告任何冲突"

    # 2. 定义模板 (JSON内部的大括号要双写)
    template = """
你是一位资深的新闻事实核查审计员，负责对融合模型的冲突处理能力进行**双层诊断审计**。

### 【输入数据】
- **Ground Truth (注入冲突标准答案)**:
{injected_block}

- **模型自汇报 (Model Self-Report)**:
{fusion_block}

- **融合正文 (Fused Output)**:
{fused_text}

---

### 【审计原则】
1. **真值唯一性**：Ground Truth 是审计的唯一标准
2. **语义等价性**：如果模型汇报的实体名与 GT 实体名在语义上指向同一事实（例如"年限"与"职业生涯时长"），应视为同一冲突，严禁仅因表述不同而判定为 FP
3. **行为优先性**：正文中的实际处理行为比自汇报更重要
4. **多数派推理规则**：
- 如果 GT 标注某版本修改了实体 X，则**未标注的版本默认保留原值**
- 例如：GT 说 "v1 将 A 改为 B"，则 v2 和 v3 默认为 A
- 多数派 = 出现次数更多的值（2:1 或 3:0）

---

## 第一阶段：冲突感知审计 (Conflict Awareness)

**目标**：判断模型是否意识到冲突的存在

对每个注入冲突，判定为以下之一：

### TP_reported（显性识别）
**定义**：模型在自汇报中明确提及该冲突

**判定标准**：
1. 模型自汇报中存在一个冲突条目
2. 该条目描述的实体与 GT 中某个冲突实体**语义匹配**
3. 语义匹配规则：
- 指向同一事实即可（如"伤亡人数"="死亡人数"）
- 不要求字面完全一致
- 如果模型用概括性描述（如"数字资产类型"）指代具体实体（如"加密货币"），只要上下文明确，即视为匹配

### TP_silent（隐性处理）
**定义**：模型未汇报，但正文中对该冲突进行了处理

**判定标准**：
1. 模型自汇报中**未提及**该冲突（不满足 TP_reported）
2. 正文中该实体的处理方式符合以下**任一情况**：
- 明确采纳了多数派值（MS）
- 明确采纳了少数派值（PF）
- 使用了模糊化策略（LN）：出现"约"、"大约"等模糊词
- 使用了删除策略（LN）：该实体在正文中被完全删除
- 使用了并列策略（LN）：正文中同时出现"A 或 B"
- 生成了幻觉值（RH）：正文值不存在于任何源
3. **排除条件**：如果正文中该实体完全缺失**且未使用模糊化替代**，判定为 FN

### FN（漏损）
**定义**：模型既未汇报，正文中也未处理

**判定标准**：
1. 模型自汇报中未提及该冲突
2. 正文中该实体完全缺失，或矛盾原样保留（同时出现多个冲突值且未说明）

### FP（误报）
**定义**：模型汇报了不存在于 GT 冲突列表中的"伪冲突"

**判定标准**：
1. 模型自汇报中存在一个冲突条目
2. 该条目描述的实体在 GT 冲突列表中**没有对应记录**
3. 即使该实体存在于源文本中，只要 GT 未标注为冲突，就是 FP

**计算公式**：
```
TP_total = TP_reported + TP_silent
Precision = TP_reported / (TP_reported + FP)
Recall = TP_total / total_injected
```

---

## 第二阶段：决策质量诊断 (Resolution Quality)

**仅针对 TP_total 中的冲突**，评估其处理策略。

### **步骤 1：推理多数派分布**

根据 Ground Truth 的冲突描述，推理出各版本的值分布：

**示例**：
```
GT: "v1 将'加密货币'改为'数字人民币'"
推理：
- v1 = "数字人民币" (1票)
- v2 = "加密货币" (默认，1票)
- v3 = "加密货币" (默认，1票)
多数派 = "加密货币" (2:1)
```

### **步骤 2：判定决策类型**

#### MS (Majority Success) - 正确决策
- ✅ 正文采纳的值 = 多数派值
- **检查点**：
 1. 从 GT 推理出多数派值
 2. 检查正文是否包含该值
 3. 确认正文未同时包含少数派值（避免并列）

**示例**：
```
多数派 = "加密货币"
正文 = "...甚至加密货币..."
判定 = MS ✅
```

#### LN (Logical Neutral) - 规避策略
- ⚠️ 正文使用模糊表达回避冲突
- **检查点**：
 - 出现模糊词："约"、"大约"、"相关"、"某"
 - 删除具体数值，仅保留定性描述
 - 使用并列结构（"A 或 B"）

**示例**：
```
多数派 = "150万"，少数派 = "200万"
正文 = "...造成巨大经济损失..."（删除数值）
判定 = LN ⚠️
```

#### RH (Reconciliation Hallucination) - 逻辑幻觉
- ❌ 正文中的值**不存在于任何源版本**
- **检查点**：
 - 取平均值（如 150 和 200 → 175）
 - 生成范围（如 "150-200"，但源中无此表述）
 - 强行并列互斥事实（用"和"连接同一属性的不同值）

**示例**：
```
v1 = "150万"，v2 = "200万"，v3 = "200万"
正文 = "...约175万..."
判定 = RH ❌（175 不存在于任何源）
```

#### PF (Propagation Failure) - 传播错误
- ❌ 正文采纳的值 = 少数派值
- ❌ 正文同时保留了多个冲突值（自相矛盾）
- **检查点**：
1. 从 GT 推理出少数派值
2. 检查正文是否包含该值
3. 确认正文未同时包含多数派值

**示例**：
```
多数派 = "加密货币" (v2, v3)
少数派 = "数字人民币" (v1)
正文 = "...甚至数字人民币..."
判定 = PF ❌
```

---

### 【特殊情况处理】

#### 情况 1：GT 只标注修改项
```
GT: "v1 将 A 改为 B"
推理：v1=B, v2=A, v3=A → 多数派=A
```

#### 情况 2：正文同时包含多个值
```
正文 = "...A 或 B..." 或 "...A 和 B..."

判定逻辑：
1. 检查 A 和 B 是否为同一属性的不同值（如"150万"vs"200万"）
2. 如果是同一属性的不同值：
- 使用"或"连接 → LN（规避，承认不确定性）
- 使用"和"连接 → PF（逻辑矛盾，同一属性不能同时有两个值）
3. 如果是不同属性（如"地震"和"洪水"）→ 不属于冲突处理，忽略
```

#### 情况 3：正文完全删除该实体
```
正文中无 A 也无 B
判定 = LN（删除策略）
```

---

### 【审计员自检清单】

在输出结果前，请逐项确认：
- [ ] 已从 GT 推理出每个冲突的完整版本分布（v1=?, v2=?, v3=?）
- [ ] 已明确标注每个冲突的多数派和少数派值
- [ ] TP_reported 判定时，已验证模型汇报的实体与 GT 实体语义匹配
- [ ] TP_silent 判定时，已确认正文中存在明确的处理行为（非完全缺失）
- [ ] MS 判定时，正文值确实等于多数派值
- [ ] PF 判定时，正文值确实等于少数派值
- [ ] RH 判定时，正文值确实不存在于任何源版本
- [ ] LN 判定时，正文确实使用了规避策略（模糊词/删除/并列）
- [ ] FP 判定时，模型汇报的冲突确实不在 GT 列表中
- [ ] 所有判定都在 reasoning 字段中给出了明确推理过程

---

### 【输出格式】(严格 JSON)
{{
"phase1_awareness": {{
"total_injected": 注入冲突总数,
"TP_reported": 0,
"TP_silent": 0,
"FN": 0,
"FP": 0,
"precision": 0.0,
"recall": 0.0
}},
"phase2_resolution": {{
"MS": 0,
"LN": 0,
"RH": 0,
"PF": 0
}},
"detailed_analysis": [
{{
"conflict_id": "冲突编号（从1开始）",
"entity_name": "GT 中的实体名称",
"reported_entity": "模型自汇报的实体名称（如果有，否则为 null）",
"mapped_gt_entity": "与模型自汇报实体最匹配的 GT 实体名称（如果有）",
"is_semantic_match": true/false,
"awareness": "TP_reported | TP_silent | FN",
"resolution": "MS | LN | RH | PF | N/A",
"reasoning": {{
"gt_distribution": "从 GT 推理的版本分布（如 v1=数字人民币, v2=加密货币, v3=加密货币）",
"majority_value": "多数派值",
"minority_value": "少数派值",
"fused_value": "正文中实际使用的值（如果有）",
 "decision_logic": "判定为 MS/LN/RH/PF 的具体原因，必须引用检查点"
}},
"evidence": "正文中的具体表述（直接引用原文）",
"note": "可选的额外观察"
}}
],
"overall_verdict": "总结模型的识别能力和决策倾向，包括：(1)显性vs隐性识别占比 (2)主要失败模式 (3)幻觉风险等级"
}}
---

### 【评估示例】

**场景**：GT = "v1 将'加密货币'改为'数字人民币'"

| 推理步骤 | 结果 |
|---------|------|
| 版本分布 | v1=数字人民币, v2=加密货币, v3=加密货币 |
| 多数派 | 加密货币 (2:1) |
| 少数派 | 数字人民币 (1:2) |

| 正文内容 | 判定 | 原因 |
|---------|------|------|
| "...加密货币..." | MS  | 采纳多数派 |
| "...数字人民币..." | PF  | 采纳少数派 |
| "...数字资产..." | LN  | 模糊化处理 |
| "...加密货币或数字人民币..." | LN  | 并列规避 |
| "...加密货币和数字人民币..." | PF  | 互斥事实并列（逻辑矛盾）|
| 完全不提该实体 | LN  | 删除策略 |

"""
    # 3. 真正执行填充
    return template.format(
        injected_block=injected_block_str,
        fusion_block=fusion_block_str,
        fused_text=fused_text,
        total_count=len(injected_conflicts)
    )
def parse_llm_evaluation_output(text: str) -> Optional[Dict]:
    if not text: return None
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try: return json.loads(text)
    except:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return None

def evaluate_single_item(item: Dict[str, Any], llm) -> Dict[str, Any]:
    fused = extract_fused_content_from_item(item)
    injected = item.get("injection_result", {}).get("conflicts", [])
    fusion_conflicts = item.get("fusion_conflict_summary", {}).get("conflicts", [])

    if not injected:
        item["conflict_detection_evaluation"] = {"error": "no_injected"}
        return item

    prompt = build_conflict_audit_prompt(injected, fusion_conflicts, fused, [])
    messages = [SystemMessage(content="你是一位严谨的第三方事实核查员。"), HumanMessage(content=prompt)]

    resp = safe_llm_invoke(llm, messages)
    if not resp:
        item["conflict_detection_evaluation"] = {"error": "llm_invoke_failed"}
        return item

    try:
        resp_text = getattr(resp, 'content', str(resp))
        evaluation = parse_llm_evaluation_output(resp_text)

        if evaluation and "phase1_awareness" in evaluation:
            p1 = evaluation["phase1_awareness"]
            p2 = evaluation["phase2_resolution"]

            # 1. 更新 Phase 1 感知统计 (区分 reported 和 silent)
            eval_stats.update(
                injected=len(injected),
                tp_rep=int(p1.get("TP_reported", 0)),
                tp_sil=int(p1.get("TP_silent", 0)),
                missed=int(p1.get("FN", 0)),
                fp=int(p1.get("FP", 0))
            )

            # 2. 更新 Phase 2 策略统计与 Recall
            eval_stats.update_diagnostic(
                ms=int(p2.get("MS", 0)),
                ln=int(p2.get("LN", 0)),
                rh=int(p2.get("RH", 0)),
                pf=int(p2.get("PF", 0)),
                recall=float(p1.get("recall", 0.0))
            )

            item["conflict_detection_evaluation"] = evaluation
    except Exception as e:
        item["conflict_detection_evaluation"] = {"error": f"logic_failed: {e}", "raw_response": resp_text[:800]}

    return item

def load_input(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])

def save_output(items, output_path, stats):
    output_data = {"evaluation_stats": stats, "total_items": len(items), "data": items}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Final results saved to {output_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--model", "-m", default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--workers", "-w", type=int, default=MAX_WORKERS)
    p.add_argument("--sample", "-s", type=int, default=0)
    return p.parse_args()
# python evaluate_conflict_zh.py -i /data1/rjj/a.test/conflict_detection/fused_data/fused_final/zh/fused_gemini-2_5-flash_zh.json -o /data1/rjj/a.test/conflict_detection/evaluation_results/gemini-2_5-flash_eval_zh.json -m gpt-5.4-pro -w 4
def main():
    args = parse_args()
    items = load_input(args.input)
    if args.sample > 0: items = items[:args.sample]

    llm = init_llm_from_config(args.model)
    actual_output_path = args.output
    if "_eval_" not in actual_output_path:
        name, ext = os.path.splitext(actual_output_path)
        # 拆解逻辑，避免 f-string 内部转义冲突
        model_tag = re.sub(r'[^\w\-]', '_', eval_stats.model_name)
        actual_output_path = f"{name}_eval_{model_tag}{ext or '.json'}"
    checkpoint = CheckpointManager(actual_output_path)
    to_process = [item for item in items if not checkpoint.is_processed(item)]
    logger.info(f"Total: {len(items)}, Already processed: {len(items)-len(to_process)}, To run: {len(to_process)}")

    with ThreadPoolExecutor(max_workers=min(args.workers, len(to_process) or 1)) as executor:
        futures = {executor.submit(evaluate_single_item, item, llm): item for item in to_process}
        iterator = tqdm(as_completed(futures), total=len(futures)) if ENABLE_PROGRESS_BAR else as_completed(futures)
        for fut in iterator:
            try:
                res = fut.result()
                checkpoint.write(res)
            except Exception as e:
                logger.error(f"Error processing item: {e}")

    # 合并结果
    final_results = []
    if os.path.exists(checkpoint.jsonl_path):
        with open(checkpoint.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f: final_results.append(json.loads(line))

    save_output(final_results, actual_output_path, eval_stats.get_stats())

if __name__ == "__main__":
    main()
