#!/usr/bin/env python3
# coding: utf-8
"""
fusion_system_with_conflict_summary.py (中文适配版)

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
import ast

# Make the repository's canonical shared configuration importable when this
# script is run directly from its track subdirectory.
COMMON_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
if COMMON_CODE_DIR not in sys.path:
    sys.path.insert(0, COMMON_CODE_DIR)

# langchain 导入
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 假设 config 模块存在并包含以下配置
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
    try:
        res = llm(messages)
    except TypeError:
        try:
            res = llm(messages=messages)
        except Exception as e:
            logger.exception("LLM 调用失败: %s", e)
            raise

    if isinstance(res, str):
        return res.strip()
    if hasattr(res, 'content') and isinstance(getattr(res, 'content'), str):
        return res.content.strip()
    return str(res).strip()

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

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 4000)
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4o-mini"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
        if api_key == "EMPTY":
            api_key = "vllm"
        self.model_stats.model_name = model_name

        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT
        )

    def _try_parse_json(self, candidate: str, log_list: List[str]) -> Optional[Dict[str, Any]]:
        if not candidate or not isinstance(candidate, str):
            return None
        try:
            return json.loads(candidate)
        except Exception:
            try:
                sanitized = sanitize_json_text(candidate)
                return json.loads(sanitized)
            except Exception:
                try:
                    lit = ast.literal_eval(candidate)
                    if isinstance(lit, dict): return lit
                except Exception:
                    pass
        return None

    def direct_fusion_basic(self, versions: List[str]) -> Dict[str, Any]:
        while len(versions) < 3:
            versions.append("")
        v1, v2, v3 = versions[:3]

        # 中文 Prompt 模板
        fusion_prompt_template = """
你是一位专业的中文新闻编辑。任务：将提供的三个版本的新闻稿整合为一篇简洁、中立且连贯的新闻文章。

严格要求：
1) 仅使用三个版本（标注为版本 1 / 版本 2 / 版本 3）中提供的信息。严禁添加、推断或引入任何外部事实、数据或时间戳。
3) 输出的第一部分必须仅包含最终整合的新闻正文（纯文本）。请勿包含任何对齐信息、JSON、注释、元数据或处理日志。
4) 请使用中文回答。正文应保持中立、简洁，适合发布。
5) 段落结构清晰，语言流利，避免重复或自相矛盾。
6) 如果某些信息仅出现在一个版本中且不详细，只要不引入新事实，仍可整合，且这些内容不计入冲突列表。

### 冲突解决策略定义：
当遇到事实矛盾（如数字、日期或地点不一致）时，必须将处理方法归类为以下四种策略之一：
- MAJORITY_VOTING（多数表决）：两个版本一致而另一个版本不同时，采用多数版本支持的值。
- RECONCILIATION（调和）：尝试合并矛盾值，如创建范围或平均值（例如：版本说10和12，则写为“10-12人”）。
- SELECTION（选择）：在没有明确多数共识的情况下，根据细节丰富度或可信度选择其中一个版本的值。
- ERROR_PROPAGATION（错误传播）：未能解决冲突，在正文中保留了矛盾值或选择了错误的少数版本。

7) 严格排除规则：冲突列表（conflicts）必须仅包含版本间相互矛盾的项。如果所有提供信息的版本都一致，严禁将其列入 JSON。如果没有任何矛盾，冲突列表可以为空。

8) 在新闻正文之后，必须在新的一行精确输出以下分隔符：
====CONFLICT_SUMMARY_JSON_START====
然后输出一个总结冲突检测与处理情况的 JSON 对象，架构如下：

{{
  "conflicts_detected": true|false,
  "conflicts": [
    {{
      "entity": "<冲突实体的文本>",
      "entity_type": "<CARDINAL|DATE|ORG|PERSON|GPE|OTHER>",
      "versions": {{
         "rewritten_v1": "<版本1中的值或null>",
         "rewritten_v2": "<版本2中的值 or null>",
         "rewritten_v3": "<版本3中的值 or null>"
      }},
      "final_decision": "<最终选择的值或处理方式，例如：保留了版本2的值“16例”>",
      "resolution_strategy": "<MAJORITY_VOTING | RECONCILIATION | SELECTION | ERROR_PROPAGATION>"
    }}, ...
  ],
  "handling": "<总体处理摘要：例如：优先考虑详细程度/多数原则/在模糊时保留版本2>"
}}

最后，在新的一行以精确的分隔符结束 JSON 部分：
====CONFLICT_SUMMARY_JSON_END====

重要提示：正文部分必须是纯文本。JSON 必须有效、可解析，且仅包含在上述分隔符之间。不要输出任何其他解释文字。

--- 版本 1 ---
<V1>

--- 版本 2 ---
<V2>

--- 版本 3 ---
<V3>

现在请生成新闻正文和冲突摘要 JSON。
"""
        fusion_prompt = fusion_prompt_template.replace("<V1>", v1).replace("<V2>", v2).replace("<V3>", v3)

        messages = [
            SystemMessage(content="你是一位专业的新闻编辑。请输出新闻正文，随后在指定分隔符内输出冲突摘要 JSON。"),
            HumanMessage(content=fusion_prompt)
        ]

        try:
            self.model_stats.increment()
            raw = call_chat_model_and_get_text(self.llm, messages)
            if not raw: raise RuntimeError("模型返回空响应")

            fused_text = raw
            conflict_json = None
            processing_log = []

            start_marker = "====CONFLICT_SUMMARY_JSON_START===="
            end_marker = "====CONFLICT_SUMMARY_JSON_END===="

            if start_marker in raw and end_marker in raw:
                parts = raw.split(start_marker, 1)
                fused_text = parts[0].strip()
                tail = parts[1]
                if end_marker in tail:
                    json_text = tail.split(end_marker, 1)[0].strip()
                    parsed = self._try_parse_json(json_text, processing_log)
                    if parsed:
                        conflict_json = parsed
                        processing_log.append("通过分隔符解析成功")

            if conflict_json is None:
                candidate, reason = _extract_json_by_brace_matching(raw)
                if candidate:
                    parsed = self._try_parse_json(candidate, processing_log)
                    if parsed:
                        conflict_json = parsed
                        fused_text = raw.replace(candidate, "").strip()
                        processing_log.append("通过括号匹配解析成功")

            if conflict_json is None:
                conflict_json = {"conflicts_detected": False, "conflicts": [], "handling": "模型未提供可解析的摘要"}

            fused_text_clean = re.sub(r'^(正文|融合新闻|最终版本|融合结果)：', '', fused_text, flags=re.IGNORECASE).strip()

            return {
                'fused_text': fused_text_clean,
                'conflict_summary': conflict_json,
                'processing_log': processing_log
            }

        except Exception as e:
            logger.exception("融合出错: %s", e)
            return {
                'fused_text': v1 or v2 or v3,
                'conflict_summary': {"conflicts_detected": False, "conflicts": [], "handling": f"错误: {str(e)}"},
                'processing_log': [f"error:{str(e)}"]
            }

# ==================== 统一处理器与流程控制 ====================

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
# ==================== I/O 与工作流 ====================

def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    """加载新闻，适配中文嵌套结构"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"文件加载失败: {e}")
        return []

    items = data if isinstance(data, list) else data.get('data', [])
    normalized = []
    for item in items:
        new_item = item.copy()
        # 提取逻辑：优先从 injection_result 找，其次找直接字段
        inj = item.get('injection_result', {})
        if isinstance(inj, dict) and 'modified_versions' in inj:
            mv = inj['modified_versions']
            new_item['rewritten_v1'] = mv.get('rewritten_v1', '')
            new_item['rewritten_v2'] = mv.get('rewritten_v2', '')
            new_item['rewritten_v3'] = mv.get('rewritten_v3', '')
        normalized.append(new_item)
    logger.info(f"成功加载 {len(normalized)} 条数据")
    return normalized

def save_fused_news(news_list: List[Dict[str, Any]], output_path: str, stats: Dict[str, Any]):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output_data = {
        "fusion_stats": stats,
        "total_items": len(news_list),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": news_list
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 结果已保存至: {output_path}")

# 工作流节点定义省略（逻辑同原版）...
class WorkflowGraph:
    def __init__(self): self.nodes, self.edges = {}, {}
    def add_node(self, name, func): self.nodes[name] = func; self.edges.setdefault(name, [])
    def add_edge(self, src, dst): self.edges[src].append(dst)
    def run(self, start, ctx):
        q, seen = [start], set()
        while q:
            n = q.pop(0)
            if n in seen: continue
            seen.add(n)
            if n in self.nodes: ctx = self.nodes[n](ctx)
            for nxt in self.edges.get(n, []): q.append(nxt)
        return ctx

def node_load(ctx):
    ctx['news'] = load_rewritten_news(ctx['args'].input)
    if ctx['args'].sample > 0: ctx['news'] = ctx['news'][:ctx['args'].sample]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--model", "-m", default=None, help="使用的 LLM 模型")
    parser.add_argument("--sample", "-s", type=int, default=0, help="采样数量")
    parser.add_argument("--fusion-method", "-fm", default="direct")
    parser.add_argument("--similarity-model-path", "-smp", default=None)
    parser.add_argument("--gpu-device", "-gpu", default="cuda:0")
    args = parser.parse_args()

    wf = WorkflowGraph()
    wf.add_node('load', node_load)
    wf.add_node('init', node_init)
    wf.add_node('fuse', node_fuse)
    wf.add_node('save', node_save)
    wf.add_edge('load', 'init')
    wf.add_edge('init', 'fuse')
    wf.add_edge('fuse', 'save')

    print(f"🚀 开始融合处理...")
    result = wf.run('load', {'args': args})
    print(f"✅ 处理完成。输出文件: {result.get('output_path')}")

if __name__ == "__main__":
    main()
