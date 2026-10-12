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

# === Checkpoint Manager ===
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

        # Phase 1 metrics
        self.tp_reported = 0
        self.tp_silent = 0
        self.missed = 0       # FN
        self.false_positives = 0  # FP

        # Phase 2 diagnostic metrics (MS, LN, RH, PF)
        self.n_majority_success = 0         # MS
        self.n_logical_neutral = 0          # LN
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

            recall = total_tp / self.total_injected if self.total_injected > 0 else 0.0
            precision = self.tp_reported / (self.tp_reported + self.false_positives) if (self.tp_reported + self.false_positives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
                    "silent_processing_rate": round(silent_rate, 4)
                },
                "phase2_resolution_global": {
                    "MS_majority_success": self.n_majority_success,
                    "LN_logical_neutral": self.n_logical_neutral,
                    "RH_reconciliation_hallucination": self.n_reconciliation_hallucination,
                    "PF_propagation_failure": self.n_propagation_failure
                },
                "diagnostic_summary": {
                    "avg_item_recall": round(avg_recall, 4)
                }
            }
eval_stats = EvaluationStats()

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
    # 1. Prepare data
    injected_lines = [
        f"{i+1}. Entity: '{c.get('entity_text', c.get('entity', 'Unknown'))}', {c.get('version_id', 'Unknown')} changed '{c.get('original_text', '')}' to '{c.get('modified_text', '')}'"
        for i, c in enumerate(injected_conflicts)
    ]
    injected_block_str = "\n".join(injected_lines) if injected_lines else "No injected conflicts"

    fusion_lines = [
        f"{i+1}. Entity: '{fc.get('entity', 'Unknown')}', Decision: '{fc.get('final_decision', '')}'"
        for i, fc in enumerate(fusion_conflicts)
    ]
    fusion_block_str = "\n".join(fusion_lines) if fusion_lines else "The fusion model reported no conflicts"

    # 2. Define template (double curly braces inside JSON blocks)
    template = """
You are a senior news fact-checking auditor responsible for conducting a **two-phase diagnostic audit** of the fusion model's conflict-handling capability.

### [Input Data]
- **Ground Truth (Injected Conflict Reference)**:
{injected_block}

- **Model Self-Report (Fusion Model's Reported Conflicts)**:
{fusion_block}

- **Fused Output (Fusion Model's Final Text)**:
{fused_text}

---

### [Audit Principles]
1. **Ground Truth is the sole authority**: All judgments must be grounded in the GT annotations.
2. **Semantic equivalence**: If a model-reported entity name is semantically equivalent to a GT entity (e.g., "career length" vs. "years in the profession"), treat them as the same conflict. Do NOT mark as FP solely due to surface wording differences.
3. **Behavior over self-report**: Actual handling behavior in the fused text takes precedence over self-reported summaries.
4. **Majority inference rule**:
   - If GT annotates that one version modified entity X, all other versions are assumed to retain the original value by default.
   - Example: GT says "v1 changed A to B" → v2 and v3 default to A.
   - Majority = the value appearing more frequently (2:1 or 3:0).

---

## Phase 1: Conflict Awareness Audit

**Goal**: Determine whether the model was aware that a conflict existed.

For each injected conflict, assign one of the following labels:

### TP_reported (Explicit Recognition)
**Definition**: The model explicitly mentioned this conflict in its self-report.

**Criteria**:
1. A conflict entry exists in the model's self-report.
2. The entity described in that entry **semantically matches** a GT conflict entity.
3. Semantic match rules:
   - Pointing to the same underlying fact is sufficient (e.g., "casualties" = "death toll").
   - Literal string identity is NOT required.
   - If the model uses a generic description (e.g., "digital asset type") to refer to a specific entity (e.g., "cryptocurrency"), treat it as a match if the context is unambiguous.

### TP_silent (Implicit Handling)
**Definition**: The model did not report the conflict, but its fused text shows clear handling behavior.

**Criteria**:
1. The model's self-report does **NOT** mention this conflict (i.e., TP_reported does not apply).
2. The fused text handles the conflicting entity via **any** of the following strategies:
   - Explicitly adopts the majority value (MS).
   - Explicitly adopts the minority value (PF).
   - Uses a vague/hedging expression (LN): e.g., "approximately", "around", "roughly", "some".
   - Deletes the entity entirely (LN): the entity does not appear in the fused text.
   - Uses a side-by-side / disjunctive expression (LN): e.g., "A or B".
   - Generates a hallucinated value (RH): the fused value does not exist in any source version.
3. **Exclusion condition**: If the entity is completely absent from the fused text AND no hedging substitute is used, classify as FN.

### FN (Missed Detection)
**Definition**: The model neither reported the conflict nor handled it in the fused text.

**Criteria**:
1. The model's self-report does not mention this conflict.
2. The entity is completely absent from the fused text, or contradictory values appear side by side without resolution.

### FP (False Positive)
**Definition**: The model reported a conflict that does not correspond to any GT-annotated conflict.

**Criteria**:
1. A conflict entry exists in the model's self-report.
2. The entity described in that entry has **no matching record** in the GT conflict list.
3. Even if the entity appears in source texts, if GT did not annotate it as a conflict, it is an FP.

**Formulas**:
```
TP_total = TP_reported + TP_silent
Precision = TP_reported / (TP_reported + FP)
Recall = TP_total / total_injected
```

---

## Phase 2: Resolution Quality Diagnosis

**Applies only to conflicts in TP_total.** Evaluate the resolution strategy used.

### Step 1: Infer Majority Distribution

Based on the GT annotation, infer the value held by each source version:

**Example**:
```
GT: "v1 changed 'cryptocurrency' to 'digital yuan'"
Inference:
  - v1 = "digital yuan"  (1 vote)
  - v2 = "cryptocurrency" (default, 1 vote)
  - v3 = "cryptocurrency" (default, 1 vote)
Majority = "cryptocurrency" (2:1)
```

### Step 2: Classify the Resolution Strategy

#### MS (Majority Success) — Correct Decision
- ✅ The fused text adopts the majority value.
- **Checklist**:
  1. Infer the majority value from GT.
  2. Verify the fused text contains that value.
  3. Confirm the fused text does NOT simultaneously contain the minority value (to avoid misclassifying a side-by-side as MS).

**Example**:
```
Majority = "cryptocurrency"
Fused text = "...even cryptocurrency..."
Verdict = MS ✅
```

#### LN (Logical Neutral) — Evasion Strategy
- ⚠️ The fused text uses vague or evasive language to sidestep the conflict.
- **Checklist**:
  - Hedging words appear: "approximately", "around", "roughly", "certain", "some".
  - Specific value is deleted; only qualitative description remains.
  - Disjunctive structure is used: "A or B".

**Example**:
```
Majority = "1.5 million", Minority = "2 million"
Fused text = "...causing enormous economic losses..." (specific figure removed)
Verdict = LN ⚠️
```

#### RH (Reconciliation Hallucination) — Logical Hallucination
- ❌ The value in the fused text **does not exist in any source version**.
- **Checklist**:
  - Averaged value (e.g., 1.5M and 2M → 1.75M).
  - Generated range (e.g., "1.5–2 million", if no source uses this phrasing).
  - Mutually exclusive facts joined with "and" (same attribute, two different values).

**Example**:
```
v1 = "1.5 million", v2 = "2 million", v3 = "2 million"
Fused text = "...approximately 1.75 million..."
Verdict = RH ❌ (1.75M does not exist in any source)
```

#### PF (Propagation Failure) — Error Propagation
- ❌ The fused text adopts the minority value, OR retains multiple conflicting values simultaneously (self-contradiction).
- **Checklist**:
  1. Infer the minority value from GT.
  2. Verify the fused text contains that value.
  3. Confirm the fused text does NOT simultaneously contain the majority value.

**Example**:
```
Majority = "cryptocurrency" (v2, v3)
Minority = "digital yuan" (v1)
Fused text = "...even digital yuan..."
Verdict = PF ❌
```

---

### [Special Cases]

#### Case 1: GT Only Annotates the Modified Version
```
GT: "v1 changed A to B"
Inference: v1=B, v2=A, v3=A → Majority=A
```

#### Case 2: Fused Text Contains Multiple Values Simultaneously
```
Fused text = "...A or B..." or "...A and B..."

Decision logic:
1. Check whether A and B are different values for the SAME attribute (e.g., "1.5M" vs. "2M").
2. If same attribute:
   - Joined with "or" → LN (evasion; acknowledges uncertainty)
   - Joined with "and" → PF (logical contradiction; same attribute cannot have two values simultaneously)
3. If different attributes (e.g., "earthquake" and "flood") → Not a conflict handling issue; ignore.
```

#### Case 3: Entity Completely Deleted from Fused Text
```
Neither A nor B appears in fused text
Verdict = LN (deletion strategy)
```

---

### [Auditor Self-Checklist]

Before producing output, confirm each item:
- [ ] Inferred the complete version distribution (v1=?, v2=?, v3=?) for every GT conflict.
- [ ] Clearly identified majority and minority values for each conflict.
- [ ] For TP_reported: verified that the model-reported entity semantically matches the GT entity.
- [ ] For TP_silent: confirmed that the fused text contains a clear handling behavior (not a complete absence).
- [ ] For MS: the fused value truly equals the majority value.
- [ ] For PF: the fused value truly equals the minority value.
- [ ] For RH: the fused value truly does not exist in any source version.
- [ ] For LN: the fused text truly uses an evasion strategy (hedging / deletion / disjunction).
- [ ] For FP: the model-reported conflict entity truly has no match in the GT list.
- [ ] Every judgment includes an explicit reasoning chain in the `reasoning` field.

---

### [Output Format] (Strict JSON)
{{
  "phase1_awareness": {{
    "total_injected": <total number of injected conflicts>,
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
      "conflict_id": "<conflict index starting from 1>",
      "entity_name": "<entity name from GT>",
      "reported_entity": "<entity name from model self-report, or null if not reported>",
      "mapped_gt_entity": "<GT entity that best matches the model-reported entity, or null>",
      "is_semantic_match": true,
      "awareness": "TP_reported | TP_silent | FN",
      "resolution": "MS | LN | RH | PF | N/A",
      "reasoning": {{
        "gt_distribution": "<version distribution inferred from GT, e.g. v1=digital yuan, v2=cryptocurrency, v3=cryptocurrency>",
        "majority_value": "<majority value>",
        "minority_value": "<minority value>",
        "fused_value": "<value actually used in fused text, or null if absent>",
        "decision_logic": "<specific reason for MS/LN/RH/PF verdict, must reference checklist items>"
      }},
      "evidence": "<direct quote from fused text supporting the judgment>",
      "note": "<optional additional observations>"
    }}
  ],
  "overall_verdict": "<Summary of the model's detection capability and resolution tendencies, including: (1) ratio of explicit vs. implicit recognition; (2) dominant failure patterns; (3) hallucination risk level>"
}}

---

### [Evaluation Example]

**Scenario**: GT = "v1 changed 'cryptocurrency' to 'digital yuan'"

| Reasoning Step | Result |
|----------------|--------|
| Version distribution | v1=digital yuan, v2=cryptocurrency, v3=cryptocurrency |
| Majority | cryptocurrency (2:1) |
| Minority | digital yuan (1:2) |

| Fused Text Content | Verdict | Reason |
|--------------------|---------|--------|
| "...cryptocurrency..." | MS  | Adopts majority value |
| "...digital yuan..." | PF  | Adopts minority value |
| "...digital assets..." | LN  | Vague/hedging expression |
| "...cryptocurrency or digital yuan..." | LN  | Disjunctive evasion |
| "...cryptocurrency and digital yuan..." | PF  | Mutually exclusive facts conjoined (logical contradiction) |
| Entity not mentioned at all | LN  | Deletion strategy |

"""
    # 3. Fill in the template
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
    messages = [SystemMessage(content="You are a rigorous third-party fact-checking auditor."), HumanMessage(content=prompt)]

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

            # 1. Update Phase 1 awareness stats (distinguish reported vs silent)
            eval_stats.update(
                injected=len(injected),
                tp_rep=int(p1.get("TP_reported", 0)),
                tp_sil=int(p1.get("TP_silent", 0)),
                missed=int(p1.get("FN", 0)),
                fp=int(p1.get("FP", 0))
            )

            # 2. Update Phase 2 resolution stats and recall
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

def main():
    args = parse_args()
    items = load_input(args.input)
    if args.sample > 0: items = items[:args.sample]

    llm = init_llm_from_config(args.model)
    actual_output_path = args.output
    if "_eval_" not in actual_output_path:
        name, ext = os.path.splitext(actual_output_path)
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

    # Merge results
    final_results = []
    if os.path.exists(checkpoint.jsonl_path):
        with open(checkpoint.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f: final_results.append(json.loads(line))

    save_output(final_results, actual_output_path, eval_stats.get_stats())

if __name__ == "__main__":
    main()
