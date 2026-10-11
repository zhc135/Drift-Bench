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
    # 修改 max_tokens 为 12288
    max_tokens = 12288
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    if api_key == "EMPTY":
        api_key = "vllm"
    if DEBUG_MODE:
        logger.info(f"init_llm_from_config: model={model_name_cfg}, temp={temperature}, max_tokens={max_tokens}, base_url={base_url}")
   
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
    # 如果已经有 alignment 且没有错误，则直接返回（用于修复模式时跳过已成功的）
    if "alignment" in item and item["alignment"] is not None and "alignment_error" not in item:
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
            # 清除可能的错误标记
            if "alignment_error" in item:
                del item["alignment_error"]
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


def is_alignment_valid(item: Dict[str, Any]) -> bool:
    """判断条目的对齐结果是否有效（结构化输出成功）"""
    # 有 alignment 字段且不为 None，且没有 alignment_error
    if "alignment" in item and item["alignment"] is not None:
        if "alignment_error" not in item:
            return True
    return False


def fix_missing_alignments(items: List[Dict[str, Any]], llm) -> Tuple[List[Dict[str, Any]], int, int]:
    """修复缺失有效对齐的条目，返回 (修复后的列表, 修复前无效数, 修复后无效数)"""
    before_invalid = sum(1 for item in items if not is_alignment_valid(item))
    if before_invalid == 0:
        logger.info("🎉 没有发现无效对齐条目，无需修复。")
        return items, 0, 0

    logger.info(f"🔧 发现 {before_invalid} 个无效对齐条目，开始针对性修复...")
    # 只对无效条目重新处理
    items_to_fix = [item for item in items if not is_alignment_valid(item)]
    fixed_items = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items_to_fix))) as executor:
        futures = {executor.submit(align_single_item, item, llm): item for item in items_to_fix}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="修复对齐"):
            try:
                res = fut.result()
                fixed_items.append(res)
            except Exception as e:
                logger.error(f"修复条目失败: {e}")
                # 保留原条目
                fixed_items.append(futures[fut])
    # 构建新列表：保留有效条目 + 修复后的条目
    new_items = []
    for item in items:
        if is_alignment_valid(item):
            new_items.append(item)
        else:
            # 找到修复后的对应条目（通过 id 或 fused_content 匹配）
            matched = None
            for fixed in fixed_items:
                # 比较 id
                if item.get("id") is not None and fixed.get("id") == item.get("id"):
                    matched = fixed
                    break
                # 比较 fused_content 的前100字符
                orig_fused = extract_fused_content_from_item(item)
                fixed_fused = extract_fused_content_from_item(fixed)
                if orig_fused[:100] == fixed_fused[:100]:
                    matched = fixed
                    break
            if matched is not None:
                new_items.append(matched)
            else:
                # 没找到则保留原条目（可能修复失败）
                new_items.append(item)
    after_invalid = sum(1 for item in new_items if not is_alignment_valid(item))
    return new_items, before_invalid, after_invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--model", "-m", default=None, help="使用的模型名称")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS)
    parser.add_argument("--sample", "-s", type=int, default=0)
    parser.add_argument("--fix-mode", action="store_true", help="修复模式：重新对齐无效条目并更新文件")
    p = parser.parse_args()

    llm = init_llm_from_config(p.model or DEFAULT_TEACHER_MODEL)
    final_output_path = p.output

    # 修复模式分支
    if p.fix_mode:
        print("🛠️ 启动对齐修复模式...")
        # 如果输出文件存在，则加载已有数据；否则从输入文件加载（视为全新处理）
        if os.path.exists(final_output_path):
            existing_data = load_input(final_output_path)
            if not existing_data:
                print("⚠️ 输出文件存在但无法加载数据，将从输入文件加载")
                existing_data = load_input(p.input)
        else:
            print("📄 输出文件不存在，将从输入文件加载数据并执行对齐")
            existing_data = load_input(p.input)
        
        if not existing_data:
            print("❌ 无法加载任何数据，请检查输入文件")
            sys.exit(1)
        
        if p.sample > 0:
            existing_data = existing_data[:p.sample]
            print(f"🔬 采样处理前 {p.sample} 篇新闻")
        
        fixed_data, before, after = fix_missing_alignments(existing_data, llm)
        print("\n" + "="*50)
        print("✅ 修复阶段执行完成！")
        print(f"📊 修复前 - 无效对齐条目数量: {before}")
        print(f"📊 修复后 - 无效对齐条目数量: {after}")
        print(f"📈 本次成功补充修正数量: {before - after}")
        print("="*50 + "\n")
        # 保存修复后的完整数据
        save_output(fixed_data, final_output_path, alignment_stats.get_stats())
        sys.exit(0)

    # 正常模式（断点续传）
    checkpoint = CheckpointManager(final_output_path)

    raw_items = load_input(p.input)
    if p.sample > 0:
        raw_items = raw_items[:p.sample]

    items_to_process = []
    processed_results = []

    if os.path.exists(checkpoint.jsonl_path):
        processed_results = load_input(checkpoint.jsonl_path)

    for item in raw_items:
        if not checkpoint.is_processed(item):
            items_to_process.append(item)

    logger.info(f"统计: 总计 {len(raw_items)}, 跳过已完成 {len(checkpoint.processed_keys)}, 待处理 {len(items_to_process)}")

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