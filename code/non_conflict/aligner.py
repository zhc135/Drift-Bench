#!/usr/bin/env python3
# coding: utf-8

import hashlib
import os
import sys
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

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

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# === 核心改进：断点续传管理器 (支持从旧结果中恢复) ===
class CheckpointManager:
    def __init__(self, output_path: str):
        # 临时进度保存为 .jsonl 文件
        self.jsonl_path = output_path + ".l" if not output_path.endswith(".l") else output_path
        self.lock = Lock()
        self.processed_keys = set()

        if os.path.exists(self.jsonl_path):
            logger.info(f"Checking existing checkpoint: {self.jsonl_path}")
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        key = self.get_stable_key(data)
                        if key:
                            self.processed_keys.add(key)
                    except:
                        continue
            logger.info(f"Checkpoint loaded: {len(self.processed_keys)} items already processed.")

    def get_stable_key(self, item: Dict) -> str:
        """
        获取稳定的 Key 以便跨进程恢复。
        逻辑：优先使用 id，其次提取 fused 文本的前100个非空白字符。
        """
        # 1. 尝试使用显式 ID
        item_id = str(item.get("id", "")).strip()
        if item_id and item_id != "None":
            return f"ID_{item_id}"

        # 2. 提取纯净文本内容 (剥离 ALIGNMENT 分隔符)
        text = extract_fused_content_from_item(item)
        if not text:
            return ""

        # 3. 产生指纹：移除所有空白符，取前100位
        # 这种方式即便之前是用 hash() 生成的，只要文本内容一致，新旧逻辑就能匹配上
        fingerprint = "".join(text.split())[:100]
        return f"FP_{fingerprint}"

    def is_processed(self, item: Dict) -> bool:
        return self.get_stable_key(item) in self.processed_keys

    def write_record(self, item: Dict):
        """线程安全地写入单行结果"""
        with self.lock:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()

ALIGNMENT_SEPARATOR = "====ALIGNMENT===="

class AlignmentStats:
    def __init__(self):
        self.model_name = ""
        self.call_count = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.call_count += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "model_name": self.model_name,
                "total_calls": self.call_count
            }

alignment_stats = AlignmentStats()

def init_llm_from_config(model_name: str):
    cfg = get_model_deployment_config(model_name)
    model_name_cfg = cfg.get("model_name", cfg.get("model", model_name))
    temperature = cfg.get("temperature", 0.0)
    max_tokens = cfg.get("max_tokens", 4000)
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    if api_key == "EMPTY":
        api_key = "vllm"
    if DEBUG_MODE:
        logger.info(f"init_llm_from_config: model={model_name_cfg}, temp={temperature}, base_url={base_url}")

    alignment_stats.model_name = model_name_cfg

    return ChatOpenAI(
        model_name=model_name_cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=base_url,
        request_timeout=REQUEST_TIMEOUT,
        max_retries=API_RETRY_LIMIT
    )

def safe_llm_invoke(llm, messages: List[Any], max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries + 1):
        try:
            alignment_stats.increment()
            resp = llm.invoke(messages)
            return resp
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/{max_retries+1}): {e}")
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
            else:
                logger.error("LLM invoke exhausted retries.")
                return None

def extract_alignment_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    if ALIGNMENT_SEPARATOR in text:
        parts = text.split(ALIGNMENT_SEPARATOR, 1)
        json_part = parts[1].strip()
        try:
            return json.loads(json_part)
        except Exception:
            m = re.search(r'(\[\s*\{.*\}\s*\])', json_part, re.DOTALL)
            if m:
                try: return json.loads(m.group(1))
                except Exception: pass
            m2 = re.search(r'(\{.*\})', json_part, re.DOTALL)
            if m2:
                try:
                    parsed = json.loads(m2.group(1))
                    if isinstance(parsed, dict): return [parsed]
                except Exception: pass
    else:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list): return parsed
            elif isinstance(parsed, dict): return [parsed]
        except Exception:
            m = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)
            if m:
                try: return json.loads(m.group(1))
                except Exception: pass
    return None

def build_alignment_prompt(fused_text: str, versions: List[str]) -> str:
    example_alignment = """
Example output format:
====ALIGNMENT====
[
  {
    "para_idx": 1,
    "text": "Example paragraph A, i.e., the complete text of the first paragraph in the fused article.",
    "sources": [1, 2],
    "evidence": [
     "1":["Excerpt from the corresponding paragraph or sentence in version 1...", and more..."],
      "2":["Excerpt from the corresponding paragraph or sentence in version 2...", and more..."]
    ],
    "conflict_resolved": "If there is a factual conflict in this paragraph between versions, briefly explain the conflict, state which version was selected in the fused article, or explain that no conflict was detected."
  },
  {
    "unused_content": {
      "1": ["Unused excerpt from version 1..."],
      "2": [],
      "3": []
    }
  }
]
"""
    prompt = f"""
You are a senior news editor/alignment specialist. Your task is: for each paragraph in the following 'Fused Article', align it with the three provided rewrite versions (Version 1 / Version 2 / Version 3), and output a JSON array describing the source and evidence for each paragraph.

Strict requirements (**very important**):
1) You may only use information from the three provided rewrite versions as evidence. Do NOT add external facts or extra information.
2) Output format must strictly be: one line writing the separator {ALIGNMENT_SEPARATOR}, followed by a JSON array.
3) Each array element MUST contain: para_idx, text, sources, evidence, conflict_resolved.
4) **unused_content**: After processing all fused article paragraphs, add a FINAL object with ONLY the key "unused_content".
5) Output JSON must be strictly valid.
6) Paragraph splitting rule: Prioritize splitting according to original paragraph boundaries in the fused article.
7) All matching content from the fused article must be supported by relevant matching evidence.
8) Do NOT output extra explanations.

answer in the same language as the input.

---- FUSED ARTICLE ----
{fused_text}

---- VERSION 1 ----
{versions[0] if len(versions) > 0 else ""}

---- VERSION 2 ----
{versions[1] if len(versions) > 1 else ""}

---- VERSION 3 ----
{versions[2] if len(versions) > 2 else ""}

{example_alignment}

Please begin and output ONLY the separator and the JSON array.
"""
    return prompt

def extract_fused_content_from_item(item: Dict[str, Any]) -> str:
    fused = item.get("fused_content", "") or item.get("fused", "")
    if not fused.strip():
        if "data" in item and isinstance(item["data"], list) and len(item["data"]) > 0:
            for data_item in item["data"]:
                if isinstance(data_item, dict):
                    inner_fused = data_item.get("fused_content", "") or data_item.get("fused", "")
                    if inner_fused.strip():
                        fused = inner_fused
                        break
    if ALIGNMENT_SEPARATOR in fused:
        fused = fused.split(ALIGNMENT_SEPARATOR)[0]
    return fused.strip()

def align_single_item(item: Dict[str, Any], llm) -> Dict[str, Any]:
    if "alignment" in item or "alignment_error" in item:
        return item

    fused = extract_fused_content_from_item(item)
    if not fused.strip():
        item["alignment_error"] = "missing_fused_content"
        return item

    versions = []
    for k in ("rewritten_v1", "rewritten_v2", "rewritten_v3"):
        version_text = item.get(k, "") or ""
        if not version_text.strip() and "data" in item and isinstance(item["data"], list):
            for data_item in item["data"]:
                if isinstance(data_item, dict) and k in data_item:
                    version_text = data_item.get(k, "")
                    break
        versions.append(version_text if version_text else "")

    while len(versions) < 3: versions.append("")

    prompt = build_alignment_prompt(fused, versions)
    messages = [
        SystemMessage(content="System: You are a meticulous text alignment assistant. Output the ALIGNMENT JSON according to the requirements."),
        HumanMessage(content=prompt)
    ]

    resp = safe_llm_invoke(llm, messages, max_retries=API_RETRY_LIMIT, base_delay=1.0)
    if not resp:
        item["alignment_error"] = "llm_invoke_failed"
        return item

    try:
        resp_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        resp_text = str(resp)

    alignment = extract_alignment_from_text(resp_text)
    if alignment is not None:
        try:
            json_str = json.dumps(alignment, ensure_ascii=False, indent=2)
            item["fused_content"] = fused.rstrip() + "\n" + ALIGNMENT_SEPARATOR + "\n" + json_str
            item["alignment"] = alignment
            item["_alignment_processing_log"] = ["success", "parsed_and_attached_alignment"]
        except Exception as e:
            item["_alignment_processing_log"] = ["success_parsed_but_attach_failed", str(e)]
            item["raw_alignment"] = resp_text
    else:
        item["_alignment_processing_log"] = ["parse_failed"]
        item["raw_alignment"] = resp_text
        item["alignment_error"] = "parse_failed"

    return item

def load_input(file_path: str) -> List[Dict[str, Any]]:
    items = []
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if file_path.endswith(('.jsonl', '.l')):
            for line in lines:
                if line.strip():
                    try: items.append(json.loads(line))
                    except: continue
            return items
        content = "".join(lines).strip()
        data = json.loads(content)
        if isinstance(data, list): return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list): return data["data"]
            return [data]
    except Exception as e:
        logger.error(f"load_input失败: {e}")
    return items

def add_model_suffix_to_filename(file_path: str, model_name: str) -> str:
    safe_model_name = re.sub(r'[^\w\-]', '_', model_name)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_parts = os.path.splitext(base_name)
    return os.path.join(dir_name, f"{name_parts[0]}_{safe_model_name}.json")

def save_output(news_list: List[Dict[str, Any]], output_path: str, stats: Optional[Dict[str, Any]] = None):
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_data = {
            "alignment_stats": stats or {},
            "alignment_model": alignment_stats.model_name or "unknown",
            "total_items": len(news_list),
            "data": news_list
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e: logger.error(f"保存失败: {e}")

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input", "-i", required=True)
    args.add_argument("--output", "-o", required=True)
    args.add_argument("--model", "-m", default=None)
    args.add_argument("--workers", "-w", type=int, default=MAX_WORKERS)
    args.add_argument("--sample", "-s", type=int, default=0)
    p = args.parse_args()
# python aligner.py --input /data1/rjj/a.test/main_experience/fused/zh/fused_zh_deepseek-v3.1.json --output /data1/rjj/a.test/main_experience/aligned/zh/aligned_zh_deepseek-v3.1.json --model gpt-oss-120b
    llm = init_llm_from_config(p.model or DEFAULT_TEACHER_MODEL)
    final_output_path = add_model_suffix_to_filename(p.output, alignment_stats.model_name or p.model)

    # 初始化断点管理器
    checkpoint = CheckpointManager(final_output_path)

    # 1. 加载所有输入
    raw_items = load_input(p.input)
    if p.sample > 0: raw_items = raw_items[:p.sample]

    # 2. 过滤掉已处理的项 (使用稳定 Key)
    items_to_process = []
    processed_results = []

    # 预加载已有的进度到内存 (为了最后保存完整的 JSON)
    if os.path.exists(checkpoint.jsonl_path):
        processed_results = load_input(checkpoint.jsonl_path)

    for item in raw_items:
        if not checkpoint.is_processed(item):
            items_to_process.append(item)

    logger.info(f"统计: 总计 {len(raw_items)}, 跳过已完成 {len(checkpoint.processed_keys)}, 待处理 {len(items_to_process)}")

    # 3. 开始批处理并实时写入
    if items_to_process:
        with ThreadPoolExecutor(max_workers=min(p.workers, len(items_to_process))) as executor:
            futures = {executor.submit(align_single_item, item, llm): item for item in items_to_process}
            iterator = tqdm(as_completed(futures), total=len(futures)) if ENABLE_PROGRESS_BAR else as_completed(futures)

            for fut in iterator:
                try:
                    res = fut.result()
                    checkpoint.write_record(res)
                    processed_results.append(res)
                except Exception as e:
                    logger.error(f"align task failed: {e}")

    save_output(processed_results, final_output_path, alignment_stats.get_stats())
    logger.info(f"任务结束，结果保存在: {final_output_path}")

if __name__ == "__main__":
    main()