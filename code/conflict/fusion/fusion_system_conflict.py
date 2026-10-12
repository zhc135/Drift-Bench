#!/usr/bin/env python3
# coding: utf-8
"""
fusion_system_with_conflict_summary.py

功能：
- 将三版重写文本融合为一篇新闻（由 LLM 生成）。
- 额外要求 LLM 输出是否识别到版本间冲突、冲突详情以及处理方式（以 JSON 输出在特定分隔符之后）。
- 将融合结果与冲突摘要保存在输出 JSON 中。
"""

import json
import logging
import os
import sys
import argparse
import re
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from threading import Lock
import ast  # 新增：用于回退解析 Python 风格字面量

# Make the repository's canonical shared configuration importable when this
# script is run directly from its track subdirectory.
COMMON_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
if COMMON_CODE_DIR not in sys.path:
    sys.path.insert(0, COMMON_CODE_DIR)

# langchain 导入（使用新版 langchain_openai）
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from config import (
    get_model_deployment_config, DEFAULT_STUDENT_MODEL,
    MAX_WORKERS, BATCH_SIZE, REQUEST_TIMEOUT, LOG_LEVEL, DEBUG_MODE,
    ENABLE_PROGRESS_BAR
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class FusionMethod:
    DIRECT = "direct"

# ==================== 模型使用统计类 ====================

class FusionModelStats:
    """融合模型使用情况统计"""
    def __init__(self):
        self.model_name = ""
        self.call_count = 0
        self.lock = Lock()
    def increment(self):
        with self.lock:
            self.call_count += 1
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {"model_name": self.model_name, "total_calls": self.call_count}
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 融合模型使用统计摘要")
        print("="*60)
        print(f"🤖 使用模型: {stats['model_name']}")
        print(f"📞 总API调用次数: {stats['total_calls']}")
        print("="*60)

# ==================== 模型使用统计类 ====================

class FusionModelStats:
    """融合模型使用情况统计"""
    def __init__(self):
        self.model_name = ""
        self.call_count = 0
        self.lock = Lock()
    def increment(self):
        with self.lock:
            self.call_count += 1
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {"model_name": self.model_name, "total_calls": self.call_count}
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 融合模型使用统计摘要")
        print("="*60)
        print(f"🤖 使用模型: {stats['model_name']}")
        print(f"📞 总 API 调用次数: {stats['total_calls']}")
        print("="*60)

# ==================== 辅助 JSON 修复函数 ====================

class CheckpointManager:
    def __init__(self, output_path: str):
        # 自动生成对应的 .jsonl 路径
        self.jsonl_path = output_path.replace('.json', '') + ".jsonl"
        self.lock = Lock()
        self.processed_fingerprints = set()

        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.processed_fingerprints.add(self._get_fingerprint(data))
                    except:
                        continue
            logger.info(f"检测到断点：已跳过 {len(self.processed_fingerprints)} 条数据")

    def _get_fingerprint(self, item: Dict) -> str:
        # 使用三个重写版本的内容拼接后取哈希作为唯一标识
        text = "".join([item.get(k, "") or "" for k in ["rewritten_v1", "rewritten_v2", "rewritten_v3"]])
        return str(hash(text))

    def is_processed(self, item: Dict) -> bool:
        return self._get_fingerprint(item) in self.processed_fingerprints

    def write(self, item: Dict):
        with self.lock:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def sanitize_json_text(text: str) -> str:
    """修复非严格 JSON：转换引号、删除尾随逗号及注释"""
    if not isinstance(text, str):
        return text
    s = text
    s = s.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r'//.*?(?=\n|$)', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = ''.join(ch for ch in s if ch >= ' ' or ch == '\n' or ch == '\t')
    s = re.sub(r',\s*(?=[}\]])', '', s)
    if s.startswith('\ufeff'):
        s = s.lstrip('\ufeff')
    return s

# ==================== LLM 调用兼容器 ====================

def call_chat_model_and_get_text(llm, messages: List[Any]) -> str:
    """
    兼容不同 langchain/langchain_openai 版本的返回格式，
    尝试以多种方式提取模型返回的文本。
    """
    try:
        # 有的版本支持直接可调用对象
        res = llm(messages)
    except TypeError:
        try:
            res = llm(messages=messages)
        except Exception as e:
            logger.exception("LLM 调用失败（位置/关键字参数都尝试过）: %s", e)
            raise

    # 字符串
    if isinstance(res, str):
        return res.strip()

    # 若为简单消息对象
    if hasattr(res, 'content') and isinstance(getattr(res, 'content'), str):
        return res.content.strip()

    # LLMResult 风格
    if hasattr(res, 'generations'):
        texts = []
        try:
            for gen_list in res.generations:
                for gen in gen_list:
                    txt = getattr(gen, 'text', None)
                    if not txt and hasattr(gen, 'message') and getattr(gen.message, 'content', None):
                        txt = gen.message.content
                    if txt:
                        texts.append(txt)
            return "\n".join(texts).strip()
        except Exception:
            pass

    # OpenAI-like .choices
    if hasattr(res, 'choices'):
        try:
            parts = []
            for c in res.choices:
                if isinstance(c, dict) and 'message' in c:
                    parts.append(c['message'].get('content', ''))
                else:
                    parts.append(str(c))
            return "\n".join(parts).strip()
        except Exception:
            pass

    # 列表类型
    if isinstance(res, list) and res:
        try:
            first = res[0]
            if hasattr(first, 'content'):
                return first.content.strip()
            return str(first).strip()
        except Exception:
            pass

    return str(res).strip()

# ---------- JSON 提取辅助 ----------
def _extract_json_by_brace_matching(text: str) -> Tuple[Optional[str], str]:
    if not text or not isinstance(text, str):
        return None, "empty_or_not_str"
    start = text.find('{')
    if start == -1:
        return None, "no_brace"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '"' and not escape:
            in_string = not in_string
        elif ch == '\\' and not escape:
            escape = True
            continue
        else:
            escape = False
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate, "ok"
                except Exception:
                    return candidate, "invalid_json"
    return None, "no_matching"

# ==================== 直接融合代理 ====================

class DirectFusionAgent:
    def __init__(self, model_config: Dict[str, Any], model_stats: FusionModelStats):
        self.model_config = model_config
        self.model_name = model_config.get("model", "unknown")
        self.model_stats = model_stats
        self.llm = self._initialize_llm(model_config)
        if DEBUG_MODE:
            logger.info(f"init DirectFusionAgent model={self.model_name}")

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.2)
        max_tokens = model_config.get("max_tokens", 4000)
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4o-mini"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
        if api_key == "EMPTY":
            api_key = "vllm"
        self.model_stats.model_name = model_name

        # 注意 langchain_openai ChatOpenAI 的构造参数在不同版本可能差异较大
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT
        )

    def _try_parse_json(self, candidate: str, log_list: List[str]) -> Optional[Dict[str, Any]]:
        """
        多策略尝试解析 candidate JSON 文本：
        1) 直接 json.loads
        2) sanitize_json_text 后 json.loads
        3) ast.literal_eval（回退，能解析单引号等 Python 字面量）
        返回解析后的 dict 或 None，并在 log_list 中追加解析步骤信息
        """
        if not candidate or not isinstance(candidate, str):
            log_list.append("_try_parse_json_no_candidate")
            return None

        # 1) 直接尝试
        try:
            obj = json.loads(candidate)
            log_list.append("json_loads_ok_direct")
            return obj
        except Exception as e:
            log_list.append(f"json_loads_failed:{str(e)}")

        # 2) sanitize 并再试
        try:
            sanitized = sanitize_json_text(candidate)
            try:
                obj = json.loads(sanitized)
                log_list.append("json_loads_ok_after_sanitize")
                return obj
            except Exception as e2:
                log_list.append(f"json_loads_failed_after_sanitize:{str(e2)}")
        except Exception as e:
            log_list.append(f"sanitize_error:{str(e)}")

        # 3) ast.literal_eval 回退（解析 Python 风格的字面量）
        try:
            lit = ast.literal_eval(candidate)
            if isinstance(lit, (dict, list)):
                log_list.append("ast_literal_eval_ok")
                return lit
            else:
                log_list.append("ast_literal_eval_not_dict_or_list")
        except Exception as e:
            log_list.append(f"ast_literal_eval_failed:{str(e)}")

        # 4) 再用 sanitize 后 ast.literal_eval
        try:
            sanitized2 = sanitize_json_text(candidate)
            lit2 = ast.literal_eval(sanitized2)
            if isinstance(lit2, (dict, list)):
                log_list.append("ast_literal_eval_ok_after_sanitize")
                return lit2
            else:
                log_list.append("ast_literal_eval_after_sanitize_not_dict_or_list")
        except Exception as e:
            log_list.append(f"ast_literal_eval_failed_after_sanitize:{str(e)}")

        return None

    def direct_fusion_basic(self, versions: List[str]) -> Dict[str, Any]:
        """
        使用 LLM 融合三版文本，并请模型额外输出冲突检测与处理的 JSON（在明确分隔符之间）。
        返回字典包含：
            fused_text: 最终融合的正文（纯文本）
            conflict_summary: dict（解析后的 JSON），若模型未输出则为 None 并在 processing_log 给出说明
            processing_log: list
        """
        while len(versions) < 3:
            versions.append("")
        v1, v2, v3 = versions[:3]

#
        # Prompt：要求模型先输出融合后的文章（纯文本），随后输出分隔符及 JSON 冲突摘要
        fusion_prompt_template = """
You are a professional news editor. Task: Integrate the three provided versions of a news story into one concise, neutral, and coherent news article.

Strict requirements:
1) Only use information present in the three versions (labeled Version 1 / Version 2 / Version 3). Do not add, infer, or introduce any external facts, data, or timestamps.
3) The output must consist solely of the final integrated news article (pure text). Do not include any alignment information, JSON, comments, meta-data, or processing logs.
4) Answer in English. The text should be neutral, concise, and suitable for publication.
5) Paragraph structure should be clear, sentences fluent, and repetition or contradictory statements avoided.
6) If information appears in only one version and is not detailed, it can still be integrated on the condition that no new facts are introduced, and these contents are not added to the conflict list, meaning they are not considered a conflict.
### Conflict Resolution Strategy Definitions:
When encountering factual contradictions (e.g., conflicting numbers, dates, or locations), you must categorize your handling method into one of these four strategies:
- MAJORITY_VOTING: Two versions agree on a fact while one differs; you adopt the value supported by the majority.
- RECONCILIATION: You attempt to merge conflicting values by creating a compromise or an average (e.g., '10-12 people' when sources say 10 and 12).
- SELECTION: You pick one version's value based on perceived detail or credibility, without a clear majority consensus.
- ERROR_PROPAGATION: You fail to resolve the conflict, either by including multiple contradictory values in the text or by selecting the incorrect minority version.
7) The first part of the output MUST be the final integrated article text ONLY (no JSON, no explanations).
important:Strict Exclusion Rule: The 'conflicts' list MUST ONLY include items where at least one version contradicts another. If all versions providing the information agree, DO NOT include it in the JSON. A 'conflicts' list with 0 items is perfectly acceptable if no contradictions exist.
8) After the article, output the following delimiter EXACTLY on a new line:
====CONFLICT_SUMMARY_JSON_START====
Then output a single JSON object that summarizes conflict detection and handling, following this schema:

{{
  "conflicts_detected": true|false,
  "conflicts": [
    {{
      "entity": "<text of conflicting entity>",
      "entity_type": "<CARDINAL|DATE|ORG|PERSON|GPE|OTHER>",
      "versions": {{
         "rewritten_v1": "<value_in_v1_or_null>",
         "rewritten_v2": "<value_in_v2_or_null>",
         "rewritten_v3": "<value_in_v3_or_null>"
      }},
      "final_decision": "<which value was chosen or how resolved, e.g., 'kept rewritten_v2 value \"16例\"'>",
      "resolution_strategy": "<MAJORITY_VOTING | RECONCILIATION | SELECTION | ERROR_PROPAGATION>"
    }}, ...
  ],
  "handling": "<summary of how conflicts were handled overall: e.g., 'prefer most substantiated / prefer latest version / majority rule / kept v2 when ambiguous'>"
}}

Finally, close the JSON section with the exact delimiter on a new line:
====CONFLICT_SUMMARY_JSON_END====

Important: The article section must be pure text only. The JSON must be valid, parseable, and contained between the two delimiters above. Do NOT output any other text or explanation.

answer in the same language as the input text.

--- Version 1 ---
<V1>

--- Version 2 ---
<V2>

--- Version 3 ---
<V3>

Produce the article and the conflict summary JSON now.
"""
        # 为避免与提示中大括号冲突，使用占位符并替换
        fusion_prompt = fusion_prompt_template.replace("<V1>", v1).replace("<V2>", v2).replace("<V3>", v3)

        messages = [
            SystemMessage(content="You are a professional news editor. Output article text followed by conflict summary JSON between specified delimiters."),
            HumanMessage(content=fusion_prompt)
        ]

        try:
            # 调用模型
            self.model_stats.increment()
            raw = call_chat_model_and_get_text(self.llm, messages)
            if not raw:
                raise RuntimeError("LLM returned empty response")

            # 尝试从 raw 中分离正文与 JSON（以分隔符为首选）
            fused_text = raw
            conflict_json = None
            processing_log: List[str] = []

            start_marker = "====CONFLICT_SUMMARY_JSON_START===="
            end_marker = "====CONFLICT_SUMMARY_JSON_END===="
            if start_marker in raw and end_marker in raw:
                parts = raw.split(start_marker, 1)
                fused_text = parts[0].strip()
                tail = parts[1]
                if end_marker in tail:
                    json_text = tail.split(end_marker, 1)[0].strip()
                    # 先尝试直接解析
                    parsed = self._try_parse_json(json_text, processing_log)
                    if parsed is not None:
                        conflict_json = parsed
                        processing_log.append("parsed_conflict_json_via_markers")
                    else:
                        # 如果失败，尝试在 tail 中做大括号匹配并解析（更精细）
                        candidate, reason = _extract_json_by_brace_matching(tail)
                        processing_log.append(f"brace_extract_after_marker:{reason}")
                        if candidate:
                            parsed2 = self._try_parse_json(candidate, processing_log)
                            if parsed2 is not None:
                                conflict_json = parsed2
                                processing_log.append("parsed_conflict_json_via_brace_after_marker")
                            else:
                                processing_log.append("failed_to_parse_json_in_marker_block")
                else:
                    processing_log.append("end_marker_missing_after_start")
            else:
                # 未使用分隔符，尝试大括号匹配解析 JSON（从 raw 中找）
                candidate, reason = _extract_json_by_brace_matching(raw)
                processing_log.append(f"brace_extract_no_markers:{reason}")
                if candidate:
                    parsed = self._try_parse_json(candidate, processing_log)
                    if parsed is not None:
                        conflict_json = parsed
                        # fused_text 为 raw 中去除 candidate 的部分
                        fused_text = raw.replace(candidate, "").strip()
                        processing_log.append("parsed_conflict_json_via_brace_matching")
                    else:
                        processing_log.append("failed_to_parse_json_from_brace_candidate")
                else:
                    processing_log.append("no_conflict_json_found")

            # 如果没有解析出 conflict_json，则设置默认结构（但不急于覆盖上层原始 injection_result）
            if conflict_json is None:
                conflict_json = {
                    "conflicts_detected": False,
                    "conflicts": [],
                    "handling": "no_conflict_summary_from_model"
                }

            # 清理 fused_text：删除常见前缀
            fused_text_clean = re.sub(
                r'^(fused news:|fused answer:|final version:|fused version:|example output:)',
                '',
                fused_text,
                flags=re.IGNORECASE
            ).strip()

            return {
                'fused_text': fused_text_clean,
                'conflict_summary': conflict_json,
                'processing_log': processing_log
            }

        except Exception as e:
            logger.exception("direct_fusion_basic error: %s", e)
            # 兜底：直接返回第一版本作为 fused_text，并标注错误
            return {
                'fused_text': v1 or v2 or v3 or "",
                'conflict_summary': {"conflicts_detected": False, "conflicts": [], "handling": f"error:{str(e)}"},
                'processing_log': [f"error:{str(e)}"]
            }

# ==================== 统一处理器 ====================

class UnifiedFusionProcessor:
    def __init__(self, model_name=None, local_similarity_model_path=None, gpu_device="cuda:0", fusion_method=FusionMethod.DIRECT):
        self.model_name = model_name or DEFAULT_STUDENT_MODEL
        self.model_config = get_model_deployment_config(self.model_name)
        self.fusion_method = fusion_method
        self.model_stats = FusionModelStats()
        self.agent = DirectFusionAgent(self.model_config, self.model_stats)
        self.max_workers = MAX_WORKERS
        self.batch_size = BATCH_SIZE

    def process_single_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        result_news = news.copy()
        versions = [result_news.get(k, "") or "" for k in ["rewritten_v1", "rewritten_v2", "rewritten_v3"]]

        if len([v for v in versions if v.strip()]) < 2:
            result_news["fused_content"] = versions[0] or result_news.get("input_text", "")
            result_news["fusion_method"] = self.fusion_method
            result_news["conflict_summary"] = {"conflicts_detected": False, "conflicts": [], "handling": "版本不足，未进行融合"}
            return result_news

        fusion_result = self.agent.direct_fusion_basic(versions)
        result_news["fused_content"] = fusion_result.get("fused_text", "")
        result_news["fusion_method"] = self.fusion_method
        result_news["_processing_log"] = fusion_result.get("processing_log", [])
        result_news["fusion_conflict_summary"] = fusion_result.get("conflict_summary")
        return result_news

    def process_news_batch(self, news_list: List[Dict[str, Any]], checkpoint: CheckpointManager) -> List[Dict[str, Any]]:
        # 过滤掉已经处理过的条目
        to_process = [n for n in news_list if not checkpoint.is_processed(n)]
        logger.info(f"待处理任务: {len(to_process)} / 总计: {len(news_list)}")

        if not to_process:
            return self._load_all_from_jsonl(checkpoint.jsonl_path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_news, n): n for n in to_process}
            iterator = tqdm(as_completed(futures), total=len(futures), desc="正在融合") if ENABLE_PROGRESS_BAR else as_completed(futures)

            for f in iterator:
                try:
                    result = f.result()
                    # 关键：每完成一条，立即写入 JSONL
                    checkpoint.write(result)
                except Exception as e:
                    logger.exception("条目处理失败: %s", e)

        # 处理完成后，从 JSONL 加载完整结果返回
        return self._load_all_from_jsonl(checkpoint.jsonl_path)

    def _load_all_from_jsonl(self, path: str) -> List[Dict[str, Any]]:
        results = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
        return results

# ==================== 工作流图与 I/O ====================

class WorkflowGraph:
    def __init__(self):
        self.nodes, self.edges = {}, {}
    def add_node(self, name: str, func):
        self.nodes[name] = func
        self.edges.setdefault(name, [])
    def add_edge(self, src: str, dst: str):
        self.edges.setdefault(src, []).append(dst)
    def run(self, start: str, context: Dict[str, Any]):
        q, seen = [start], set()
        while q:
            n = q.pop(0)
            if n in seen:
                continue
            seen.add(n)
            func = self.nodes.get(n)
            if func:
                context = func(context)
            for nxt in self.edges.get(n, []):
                if nxt not in seen:
                    q.append(nxt)
        return context

# load_rewritten_news：支持新的嵌套格式
def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    """
    加载重写新闻数据，支持嵌套的 injection_result 结构
    保留所有原始字段
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"load_rewritten_news open/load error: {e}", exc_info=True)
        return []

    def try_parse_json_string(s: Any) -> Any:
        """如果 s 是 JSON 字符串，则尝试解析返回对象；否则返回原始值"""
        if isinstance(s, str):
            s_strip = s.strip()
            # 尝试解析为 JSON
            if (s_strip.startswith('{') and s_strip.endswith('}')) or (s_strip.startswith('[') and s_strip.endswith(']')):
                try:
                    return json.loads(s_strip)
                except Exception:
                    # 尝试 ast.literal_eval 作为回退
                    try:
                        return ast.literal_eval(s_strip)
                    except Exception:
                        return s
        return s

    def extract_versions(item: Dict[str, Any]) -> Dict[str, str]:
        """从项目数据中提取三个版本的重写文本"""
        versions = {}

        # 检查是否有 injection_result -> modified_versions 结构
        if 'injection_result' in item and isinstance(item['injection_result'], dict):
            injection_result = item['injection_result']
            # 如果 injection_result 本身包含字符串化的 modified_versions，先尝试解析
            if isinstance(injection_result, str):
                injection_result = try_parse_json_string(injection_result)
            modified_versions = injection_result.get('modified_versions', {}) if isinstance(injection_result, dict) else {}
            if isinstance(modified_versions, str):
                modified_versions = try_parse_json_string(modified_versions)
            if isinstance(modified_versions, dict):
                versions['rewritten_v1'] = modified_versions.get('rewritten_v1', '') or ''
                versions['rewritten_v2'] = modified_versions.get('rewritten_v2', '') or ''
                versions['rewritten_v3'] = modified_versions.get('rewritten_v3', '') or ''

        # 如果没有从 injection_result 中提取到，检查直接字段
        if not any(versions.values()):
            if 'rewritten_v1' in item or 'rewritten_v2' in item or 'rewritten_v3' in item:
                versions['rewritten_v1'] = item.get('rewritten_v1', '') or ''
                versions['rewritten_v2'] = item.get('rewritten_v2', '') or ''
                versions['rewritten_v3'] = item.get('rewritten_v3', '') or ''

        # 如果还没有，尝试从 rewrite_versions 列表中提取
        if not any(versions.values()):
            rewrite_list = item.get('rewrite_versions') or item.get('top_rewrites') or item.get('rewrites') or []
            if isinstance(rewrite_list, str):
                # 有时是字符串化的列表，尝试解析
                parsed = try_parse_json_string(rewrite_list)
                if isinstance(parsed, list):
                    rewrite_list = parsed
            if isinstance(rewrite_list, list):
                for i in range(3):
                    key = f'rewritten_v{i+1}'
                    if i < len(rewrite_list):
                        entry = rewrite_list[i]
                        if isinstance(entry, dict):
                            versions[key] = entry.get('final_rewritten_text') or entry.get('final_text') or entry.get('text') or ''
                        elif isinstance(entry, str):
                            versions[key] = entry
                        else:
                            versions[key] = ''
                    else:
                        versions[key] = ''

        # 作为最后手段，尝试从 input_text 中提取（不会拆分，只作为 v1）
        if not any(versions.values()):
            versions['rewritten_v1'] = item.get('input_text', '') or ''
            versions['rewritten_v2'] = ''
            versions['rewritten_v3'] = ''

        # 确保所有三个键都存在
        for i in range(3):
            key = f'rewritten_v{i+1}'
            if key not in versions:
                versions[key] = ''

        return versions

    normalized: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        # 如果是单个字典（可能包含 data 列表）
        if 'data' in data and isinstance(data['data'], list):
            # 处理包含 data 字段的情况
            for item in data['data']:
                if isinstance(item, dict):
                    normalized_item = item.copy()  # 保留所有原始字段
                    # 添加提取的版本
                    versions = extract_versions(item)
                    normalized_item.update(versions)
                    normalized.append(normalized_item)
        elif any(k in data for k in ('rewrite_versions', 'top_rewrites', 'rewrites', 'rewritten_v1', 'injection_result')):
            # 处理单个新闻项
            normalized_item = data.copy()  # 保留所有原始字段
            versions = extract_versions(data)
            normalized_item.update(versions)
            normalized.append(normalized_item)
        else:
            # 查找包含新闻条目的列表
            found_list = None
            for k, v in data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    if any(key in v[0] for key in ('rewrite_versions', 'top_rewrites', 'rewritten_v1', 'injection_result')):
                        found_list = v
                        break

            if found_list is not None:
                for item in found_list:
                    if isinstance(item, dict):
                        normalized_item = item.copy()  # 保留所有原始字段
                        versions = extract_versions(item)
                        normalized_item.update(versions)
                        normalized.append(normalized_item)
            else:
                # 如果没有找到列表，将整个字典作为一个新闻项
                normalized_item = data.copy()  # 保留所有原始字段
                versions = extract_versions(data)
                normalized_item.update(versions)
                normalized.append(normalized_item)

    elif isinstance(data, list):
        # 处理新闻项列表
        for item in data:
            if isinstance(item, dict):
                normalized_item = item.copy()
                versions = extract_versions(item)
                normalized_item.update(versions)
                normalized.append(normalized_item)
            else:
                normalized.append({
                    'input_text': str(item) if item else '',
                    'rewritten_v1': '',
                    'rewritten_v2': '',
                    'rewritten_v3': ''
                })
    else:
        logger.error("load_rewritten_news: unexpected JSON root type: %s", type(data))
        return []

    logger.info(f"加载了 {len(normalized)} 条新闻数据")

    # 记录一些示例数据，方便调试
    if normalized and DEBUG_MODE:
        sample_item = normalized[0]
        logger.debug(f"示例数据 - 字段: {list(sample_item.keys())}")
        if 'rewritten_v1' in sample_item:
            logger.debug(f"rewritten_v1 长度: {len(sample_item['rewritten_v1'])} 字符")
        if 'injection_result' in sample_item:
            logger.debug(f"包含 injection_result 字段")

    return normalized

def add_model_suffix_to_filename(file_path: str, model_name: str) -> str:
    safe_model_name = re.sub(r'[^\w\-]', '_', model_name)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_parts = os.path.splitext(base_name)
    file_name_without_ext = name_parts[0]
    file_ext = name_parts[1] or ".json"
    new_base_name = f"{file_name_without_ext}_{safe_model_name}{file_ext}"
    return os.path.join(dir_name, new_base_name)

def save_fused_news(news_list: List[Dict[str, Any]], output_path: str, stats: Dict[str, Any]):
    """
    保存融合后的新闻，保留原始输入中的 injection_result 字段
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cleaned_news_list = []

    for news_item in news_list:
        # 创建清理后的项目，保留原始字段
        cleaned_item = {}

        # 保留原始数据的所有字段（除了我们可能会修改的字段）
        for key, value in news_item.items():
            if key not in ['fused_content', 'fusion_method', 'conflict_summary', '_processing_log']:
                cleaned_item[key] = value

        # 获取融合内容
        fused_content = news_item.get("fused_content", "")

        # 处理可能的对齐标记
        if "====ALIGNMENT====" in fused_content:
            parts = fused_content.split("====ALIGNMENT====")
            cleaned_item["fused_content"] = parts[0].strip()
        else:
            cleaned_item["fused_content"] = fused_content.strip()

        # 添加融合方法
        cleaned_item["fusion_method"] = news_item.get("fusion_method", "unknown")

        # 添加处理日志（如果存在）
        if '_processing_log' in news_item:
            cleaned_item["_processing_log"] = news_item['_processing_log']

        # 添加融合后的冲突摘要
        cleaned_item["fusion_conflict_summary"] = news_item.get("conflict_summary",
                                                              {"conflicts_detected": False,
                                                               "conflicts": [],
                                                               "handling": "missing"})

        # 保留原始的 injection_result（如果存在）

        cleaned_news_list.append(cleaned_item)

    # 构建输出数据结构
    output_data = {
        "fusion_stats": stats,
        "fusion_model": stats.get("model_name", "unknown"),
        "total_items": len(cleaned_news_list),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": cleaned_news_list
    }

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 保存 {len(cleaned_news_list)} 条融合新闻到 {output_path}")
# ===== Workflow nodes =====

def node_load(ctx):
    a = ctx['args']
    n = load_rewritten_news(a.input)
    if a.sample > 0:
        n = n[:a.sample]
    ctx['news'] = n
    logger.info(f"加载了 {len(n)} 条新闻数据")
    return ctx

def node_init(ctx):
    a = ctx['args']
    ctx['processor'] = UnifiedFusionProcessor(a.model, a.similarity_model_path, a.gpu_device, a.fusion_method)
    return ctx

def node_fuse(ctx):
    a = ctx['args']
    # 预先计算输出路径以创建断点文件
    model_tag = re.sub(r'[^\w\-]', '_', ctx['processor'].model_name)
    base_out_path = a.output
    if not base_out_path.endswith('.json'):
        base_out_path += f"_{model_tag}.json"

    # 初始化断点管理器
    checkpoint = CheckpointManager(base_out_path)

    # 传入 checkpoint 进行处理
    ctx['fused'], ctx['stats'] = ctx['processor'].process_news_batch(ctx['news'], checkpoint)
    ctx['final_json_path'] = base_out_path
    return ctx

def node_save(ctx):
    # 此时 ctx['fused'] 已经包含了从 JSONL 汇总的所有数据
    save_fused_news(ctx['fused'], ctx['final_json_path'], ctx['stats'])
    ctx['output_path'] = ctx['final_json_path']
    return ctx

def node_stats(ctx):
    f = ctx.get('fused', [])
    stats = ctx.get('stats', {})
    success_count = sum(1 for x in f if x.get('fused_content') and x.get('fused_content', '').strip())
    ctx['final_stats'] = {
        'total': len(f),
        'success': success_count,
        'model_stats': stats
    }
    if ctx.get('processor'):
        ctx['processor'].model_stats.print_summary()
    return ctx

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--model", "-m", default=None)
    p.add_argument("--similarity-model-path", "-smp", default=None)
    p.add_argument("--gpu-device", "-gpu", default="cuda:0")
    p.add_argument("--sample", "-s", type=int, default=0)
    p.add_argument("--fusion-method", "-fm", default=FusionMethod.DIRECT)
    return p.parse_args()
# python fusion_system_conflict.py --input /data1/rjj/a.test/conflict_detection/conflict_test.json --output /data1/rjj/a.test/conflict_detection/fused --model  doubao-1-5-pro-256k-250115 --fusion-method direct
def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    ctx = {'args': args}
    wf = WorkflowGraph()

    # 添加工作流节点
    wf.add_node('load', node_load)
    wf.add_node('init', node_init)
    wf.add_node('fuse', node_fuse)
    wf.add_node('save', node_save)
    wf.add_node('stats', node_stats)

    # 设置工作流边
    wf.add_edge('load', 'init')
    wf.add_edge('init', 'fuse')
    wf.add_edge('fuse', 'save')
    wf.add_edge('save', 'stats')

    # 运行工作流
    print(f"🚀 开始融合处理，输入文件: {args.input}")
    print(f"   使用模型: {args.model or DEFAULT_STUDENT_MODEL}")
    print(f"   融合方法: {args.fusion_method}")

    result = wf.run('load', ctx)

    s = result.get('final_stats', {})
    output_path = result.get('output_path', args.output)

    print(f"\n✅ 融合完成:")
    print(f"   总处理: {s.get('total',0)} 条")
    print(f"   成功融合: {s.get('success',0)} 条")
    print(f"   输出文件: {output_path}")

    model_stats = s.get('model_stats', {})
    if model_stats:
        print(f"   使用模型: {model_stats.get('model_name', 'unknown')}")
        print(f"   模型调用次数: {model_stats.get('total_calls', 0)}")

    # 显示一些输出数据示例
    if s.get('success', 0) > 0:
        print(f"\n📊 输出数据结构:")
        fused_data = result.get('fused', [])
        if fused_data:
            sample = fused_data[0]
            print(f"   示例条目字段: {list(sample.keys())}")
            if 'fused_content' in sample:
                fused_len = len(sample['fused_content'])
                print(f"   融合内容长度: {fused_len} 字符")
            if 'fusion_conflict_summary' in sample:
                conf_summary = sample['fusion_conflict_summary']
                conflicts_detected = conf_summary.get('conflicts_detected', False)
                conflicts_count = len(conf_summary.get('conflicts', []))
                print(f"   融合冲突检测: {conflicts_detected} ({conflicts_count} 个冲突)")
if __name__ == "__main__":
    main()
