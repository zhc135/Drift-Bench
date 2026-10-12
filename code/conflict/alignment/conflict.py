#!/usr/bin/env python3
# coding: utf-8
"""
inject_conflicts_langchain_modified_single_version_with_fix.py

说明：
- 随机从跨版本共享实体池中选择若干实体，每个实体只在一个版本中被修改（随机选取版本）。
- 修复实体匹配容错：在提取共享实体时尝试扩展实体 span（包含单位/千分位等），
  并在校验时采用正规化比较（去逗号/单位/空格并尝试数值比较），以避免 "3" vs "3,123人" 类不匹配警告。
- 输出为精简格式：modified_versions、conflicts、version_conflicts、summary。
"""

import os
import sys
import re
import json
import ast
import argparse
import logging
import time
import random
from typing import List, Dict, Any, Tuple, Set, Optional
from tqdm import tqdm
from collections import defaultdict

# Make the repository's canonical shared configuration importable when this
# script is run directly from its track subdirectory.
COMMON_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
if COMMON_CODE_DIR not in sys.path:
    sys.path.insert(0, COMMON_CODE_DIR)

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain / OpenAI 兼容导入
try:
    from langchain_openai import ChatOpenAI
except Exception:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except Exception:
        ChatOpenAI = None
from langchain.schema import SystemMessage, HumanMessage

# 配置导入回退
try:
    from config import get_model_deployment_config, DEFAULT_STUDENT_MODEL, REQUEST_TIMEOUT
except ImportError:
    def get_model_deployment_config(model_name):
        return {
            "model": model_name,
            "model_name": model_name,
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
            "max_tokens": int(os.environ.get("MAX_TOKENS", "1000"))
        }
    DEFAULT_STUDENT_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    REQUEST_TIMEOUT = 120

# spaCy 实体抽取（尝试加载中文和英文模型）
try:
    import spacy
    NLP_ZH = None
    NLP_EN = None
    try:
        NLP_ZH = spacy.load("zh_core_web_sm")
        logger.info("✅ 成功加载 spaCy 中文模型 (zh_core_web_sm)")
    except Exception as e:
        logger.warning(f"⚠️ 无法加载 spaCy 中文模型: {e}")

    # 尝试加载英文模型
    try:
        NLP_EN = spacy.load("en_core_web_sm")
        logger.info("✅ 成功加载 spaCy 英文模型 (en_core_web_sm)")
    except Exception as e:
        logger.warning(f"⚠️ 无法加载 spaCy 英文模型: {e}")

    # 兼容旧代码 nlp 变量
    # nlp 变量指向第一个成功的模型，但 extract_entities 会动态选择
    nlp = NLP_ZH if NLP_ZH else NLP_EN

    if not nlp:
        raise RuntimeError("所有 spaCy 模型加载失败")

except Exception as e:
    NLP_ZH = None
    NLP_EN = None
    nlp = None
    logger.warning(f"⚠️ spaCy 不可用，或模型加载失败: {e}。将无法提取实体")

# ---------- 工具函数 ----------
def extract_entities(text: str) -> List[Dict[str, Any]]:
    ents = []

    # 动态选择模型：检查文本中是否包含中文字符
    has_chinese = re.search(r'[\u4e00-\u9fff]', text)

    current_nlp = None
    # 优先使用中文模型处理中文文本
    if has_chinese and globals().get('NLP_ZH'):
        current_nlp = globals()['NLP_ZH']
    # 优先使用英文模型处理非中文文本
    elif not has_chinese and globals().get('NLP_EN'):
        current_nlp = globals()['NLP_EN']
    # 如果只有一种模型可用，就用它（例如只有中文或只有英文）
    elif globals().get('nlp'):
        current_nlp = globals()['nlp']

    if current_nlp and text:
        doc = current_nlp(text)
        # 扩展实体标签，增加 LOC 以更好地适应英文模型
        allowed_labels = ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "PERCENT", "LOC"]

        for e in doc.ents:
            if e.label_ in allowed_labels:
                ents.append({
                    "text": e.text.strip(),
                    "label": e.label_,
                    "start_char": e.start_char,
                    "end_char": e.end_char
                })
    # 去重
    seen = set()
    out = []
    for e in ents:
        key = (e["text"], e["label"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

def find_shared_entities_across_versions(versions: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    查找在所有版本中都出现的实体，并尝试扩展实体 span 以包含单位（人/年/元/%/万）或千分位数字。
    返回的实体文本尽量为“完整形式”，便于后续与LLM输出比较。
    """
    entity_occurrence = defaultdict(set)
    version_entities = {}
    for version_id, text in versions.items():
        entities = extract_entities(text)
        version_entities[version_id] = entities
        for entity in entities:
            key = (entity['text'], entity['label'])
            entity_occurrence[key].add(version_id)

    shared_entities = []
    for (ent_text, label), version_set in entity_occurrence.items():
        if version_set == set(versions.keys()):
            pos_info = {}
            full_texts = {}
            for vid in versions:
                text_full = versions[vid] or ""
                idx = text_full.find(ent_text)
                if idx == -1:
                    pos_info[vid] = -1
                    full_texts[vid] = ent_text
                else:
                    start = idx
                    end = idx + len(ent_text)
                    # 包含紧随单位字符（如 人 年 月 日 元 % 万 等）
                    if end < len(text_full) and re.match(r'[\u4e00-\u9fff％%人元年月日万]', text_full[end]):
                        end += 1
                    # 向前扩展以包含可能的千分位数字，例如 spaCy 可能把 "3,123人" 识别为 "3"
                    # 最多向前扩展 10 字符，匹配连续数字和逗号
                    expand_start = max(0, start - 10)
                    m = re.search(r'[\d,]{1,20}$', text_full[expand_start:start])
                    if m:
                        start = expand_start + m.start()
                    full_extracted = text_full[start:end]
                    pos_info[vid] = start
                    full_texts[vid] = full_extracted
            # 选取第一个非空 full_texts 作为代表文本（尽量完整）
            representative = next((v for v in full_texts.values() if v), ent_text)
            shared_entities.append({
                "text": representative,
                "label": label,
                "appears_in": list(version_set),
                "positions": pos_info
            })
    logger.info(f"🔍 找到 {len(shared_entities)} 个跨版本共享实体（含扩展）")
    return shared_entities

def locate_span(doc_text: str, span_text: str) -> Tuple[Optional[int], Optional[int]]:
    idx = doc_text.find(span_text)
    if idx == -1:
        return (None, None)
    return (idx, idx + len(span_text))

def _extract_json_by_brace_matching(text: str) -> Optional[str]:
    start = text.find('{')
    if start == -1:
        return None
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
                return text[start:i+1]
    return None

def _try_repair_json_text(txt: str) -> str:
    repaired = re.sub(r',\s*(\}|\])', r'\1', txt)
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        pass
    try:
        obj = ast.literal_eval(repaired)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    return repaired

def extract_json_from_text(text: str) -> Tuple[Optional[str], str]:
    if not text or not isinstance(text, str):
        return (None, "empty_or_not_str")
    cleaned = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.IGNORECASE)
    json_text = _extract_json_by_brace_matching(cleaned)
    if not json_text:
        return (None, "no_brace_match")
    try:
        json.loads(json_text)
        return (json_text, "ok")
    except Exception:
        repaired = _try_repair_json_text(json_text)
        try:
            json.loads(repaired)
            return (repaired, "repaired")
        except Exception as e2:
            return (None, f"json_decode_failed: {e2}")

# 比较/正规化辅助函数
def _normalize_for_compare(s: Optional[str]) -> str:
    if s is None:
        return ""
    # 去掉空白与千分位逗号，转小写
    s0 = re.sub(r'[\s,]', '', s).lower()
    # 移除常见单位后缀（人/年/月/日/元/%/万 等）
    s0 = re.sub(r'[人年月日元％%万]+$', '', s0)
    return s0

def _extract_number(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r'([\d,]+)', s)
    if not m:
        return None
    num_str = m.group(1).replace(',', '')
    try:
        return int(num_str)
    except:
        return None

# ---------- LLM 客户端 ----------
class LLMClient:
    def __init__(self, model_name: str = None, model_config: Dict[str,Any] = None, max_retries: int = 3):
        self.max_retries = max_retries
        if model_config is None:
            model_name = model_name or DEFAULT_STUDENT_MODEL
            try:
                model_config = get_model_deployment_config(model_name)
            except Exception as e:
                logger.warning(f"⚠️ 无法获取模型配置: {e}，使用默认回退")
                model_config = {
                    "model": model_name,
                    "model_name": model_name,
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    "temperature": 0.7,
                    "max_tokens": 3000
                }
        api_key = model_config.get("api_key") or model_config.get("openai_api_key", "")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY 未设置！")
        if api_key == "EMPTY":
            api_key = "vllm"
        base_url = model_config.get("base_url") or model_config.get("openai_api_base", "https://api.openai.com/v1")
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens",3000)
        model = model_config.get("model_name") or model_config.get("model", "gpt-4o-mini")
        if ChatOpenAI is None:
            raise RuntimeError("❌ 无法导入 ChatOpenAI，请安装 langchain-openai 或 langchain-community")
        try:
            self.client = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                openai_api_base=base_url,
                request_timeout=REQUEST_TIMEOUT
            )
        except Exception as e:
            logger.exception(f"❌ 初始化 LLM 失败: {e}")
            raise

    def _call_llm(self, messages: List[Any]) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.invoke(messages)
                return getattr(resp, "content", str(resp))
            except Exception as e:
                wait = (2 ** attempt) * 0.5
                logger.warning(f"⚠️ LLM 调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}，{wait:.1f}s 后重试...")
                time.sleep(wait)
        raise RuntimeError(f"❌ LLM 调用失败：超过 {self.max_retries} 次重试")

    def inject_cross_version_conflicts(
        self,
        versions: Dict[str, str],
        shared_entities: List[Dict[str, Any]],
        min_entities_to_modify: int = 1,
        max_entities_to_modify: int = 5
    ) -> Dict[str, Any]:
        # 随机选取将要修改的实体（在 min~max 之间）
        if not shared_entities:
            return self._create_no_entity_response(versions)

        max_allowed = min(max_entities_to_modify, len(shared_entities))
        min_allowed = min(min_entities_to_modify, max_allowed)
        k = random.randint(min_allowed, max_allowed) if max_allowed >= min_allowed else max_allowed
        entities_to_use = random.sample(shared_entities, k)

        # 实体信息字符串化（用于提示）
        entities_lines = []
        for i, ent in enumerate(entities_to_use, 1):
            positions = ', '.join(f"{vid}:{pos}" for vid, pos in ent['positions'].items())
            entities_lines.append(f"{i}. 实体: '{ent['text']}' (类型: {ent['label']}) → 出现在所有版本，位置: {positions}")
        entities_str = "\n".join(entities_lines)

        # 截断版本文本（避免超长）
        versions_truncated = { vid: (txt[:1200] + "..." if len(txt) > 1200 else txt) for vid, txt in versions.items() }

        # 生成 prompt（强制：每个实体只在一个版本中修改）
        system_prompt = (
            "You are a professional text consistency-breaking engineer, specializing in generating natural and stealthy single-version conflicts.\n\n"
            "Task: Given three rewritten versions and a list of shared entities, inject conflicts by modifying each selected shared entity in EXACTLY ONE of the three versions (choose which version per entity). Do NOT modify the same entity in more than one version.\n\n"
            "Core Rules (MUST follow):\n"
            "1. For each selected shared entity, apply a modification in exactly one version (no zero-modifications, no multi-version modifications for the same entity).\n"
            "2. Modifications must be natural and fluent (semantic and grammar preserved).\n"
            "3. Preserve numeric formatting (e.g., '3,123' -> '3,122', not '3122').\n"
            "4. Prioritize modifying DATE, CARDINAL, ORG, PERSON entities when practical.\n\n"
            "Suggested Modifications (examples):\n"
            " - DATE: '2023' -> '2022' or '2024'\n"
            " - CARDINAL/MONEY: '1,000,000' -> '950,000' / '1,050,000'\n"
            " - ORG: '北京大学' -> '清华大学' / '复旦大学'\n"
            " - PERSON: '张伟' -> '李强' / '王芳' (maintain role/gender consistency)\n"
            "\n"
            "STRICT PROHIBITIONS:\n"
            "- Do NOT modify non-shared entities.\n"
            "- Do NOT change punctuation or spacing unintentionally (modified output must appear verbatim in the final version string).\n"
            "- DO NOT perform the same modification for the same entity in more than one version.\n"
            "\n"
            "OUTPUT RULES:\n"
            "- Output ONLY a valid JSON object and nothing else.\n"
            "- Schema will be provided. Each conflict entry must indicate which version was modified (version_id) and show original_text and modified_text.\n"
        )

        # 更简洁的输出 schema（stringified for prompt）
        output_schema = {
            "type": "object",
            "properties": {
                "modified_versions": {
                    "type": "object",
                    "properties": {
                        "rewritten_v1": {"type": "string"},
                        "rewritten_v2": {"type": "string"},
                        "rewritten_v3": {"type": "string"}
                    },
                    "required": ["rewritten_v1", "rewritten_v2", "rewritten_v3"]
                },
                "conflict_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_text": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "version_id": {"type": "string"},
                            "original_text": {"type": "string"},
                            "modified_text": {"type": "string"},
                            "modification_type": {"type": "string"}
                        },
                        "required": ["entity_text", "entity_type", "version_id", "original_text", "modified_text", "modification_type"]
                    },
                    "minItems": 1
                }
            },
            "required": ["modified_versions", "conflict_entities"]
        }

        user_prompt = (
            "Please perform the following operations:\n\n"
            "【Three version texts】\n"
            + "\n---\n".join(f"【{vid}】\n{text}" for vid, text in versions_truncated.items())
            + f"\n\n【Selected shared entities (total: {len(entities_to_use)})】\n{entities_str}\n\n"
            "【Modification requirements】\n"
            "- For each listed entity, modify it in EXACTLY ONE of the three versions (choose the version randomly/appropriately).\n"
            "- Do NOT modify that entity in the other two versions.\n"
            "- Modifications must be fluent and context-appropriate.\n\n"
            "【Critical examples — MUST FOLLOW】\n"
            "- If original is '3,123人', correct modification: '3,122人'; incorrect: '3122人'.\n"
            "- Every 'modified_text' must appear EXACTLY (verbatim) in its corresponding version text field in 'modified_versions'.\n"
            "\n"
            "【Output format】(Strictly follow the schema below):\n"
            f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n\n"
            "⚠️ FINAL REMINDERS:\n"
            "- NEVER output non-JSON content.\n"
            "- Each entity must have exactly one version modified.\n"
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            raw_text = self._call_llm(messages)
            json_text, reason = extract_json_from_text(raw_text)
            if not json_text:
                raise ValueError(f"LLM 未返回合法 JSON（reason: {reason}）\nRaw: {raw_text[:300]}...")
            result = json.loads(json_text)
            # 验证
            self._validate_llm_output(result, versions, entities_to_use)
            return result
        except Exception as e:
            logger.error(f"❌ LLM 输出解析/校验失败: {e}")
            return self._create_fallback_response(versions, entities_to_use)

    def _validate_llm_output(self, result: Dict[str, Any], original_versions: Dict[str, str], shared_entities: List[Dict]):
        """
        严格校验 LLM 输出：
        - 包含三个版本文本
        - 每个 conflict_entity 必须在 exactly one version（version_id）
        - modified_text 必须出现在对应 modified_versions[version_id] 中
        - original_text 与 shared entity 的 entity_text 进行容错比较（正规化或数值比较）
        """
        mod_vers = result.get("modified_versions", {})
        conflicts = result.get("conflict_entities", [])

        # 1. 必须含三个版本
        for vid in ["rewritten_v1", "rewritten_v2", "rewritten_v3"]:
            if vid not in mod_vers:
                raise ValueError(f"缺失版本 {vid}")
            if not isinstance(mod_vers[vid], str):
                raise ValueError(f"版本 {vid} 非字符串")

        # 2. 每个 conflict_entity: 必须有 version_id 且字段齐全
        for ent in conflicts:
            entity_text = ent.get("entity_text")
            ent_type = ent.get("entity_type")
            version_id = ent.get("version_id")
            original_text = ent.get("original_text")
            modified_text = ent.get("modified_text")
            if not all([entity_text, ent_type, version_id, original_text, modified_text]):
                raise ValueError(f"实体条目缺少字段: {ent}")
            if version_id not in ["rewritten_v1", "rewritten_v2", "rewritten_v3"]:
                raise ValueError(f"非法 version_id: {version_id}")

            # 容错比较 original_text 与 entity_text
            n_orig = _normalize_for_compare(original_text)
            n_ent = _normalize_for_compare(entity_text)
            if n_orig != n_ent and not (entity_text in original_text or original_text in entity_text):
                # 尝试数值比较（若两者包含数字）
                num_orig = _extract_number(original_text)
                num_ent = _extract_number(entity_text)
                if num_orig is not None and num_ent is not None and num_orig == num_ent:
                    pass  # 数值上等价
                else:
                    # 仅记录警告，不抛错；后续 post_validate 会尝试容错替换
                    logger.warning(f"⚠️ 实体 '{entity_text}' 的 original_text='{original_text}' 与 entity_text 不匹配（规范化后: '{n_ent}' vs '{n_orig}'）")

            # 检查 modified_text 是否出现在对应版本文本中
            if modified_text not in mod_vers[version_id]:
                raise ValueError(f"版本 {version_id} 的输出文本中未找到修改后内容: '{modified_text}'")

        logger.info(f"✅ LLM 输出校验通过：{len(conflicts)} 个冲突实体（单版本修改）")

    def _create_no_entity_response(self, versions: Dict[str, str]) -> Dict[str, Any]:
        return {
            "modified_versions": versions.copy(),
            "conflict_entities": [],
            "warning": "无共享实体，无法注入冲突"
        }

    def _create_fallback_response(self, versions: Dict[str, str], shared_entities: List[Dict]) -> Dict[str, Any]:
        """
        本地 fallback：对随机选取的实体在随机的单个版本做简单扰动（遵循单版本修改规则）。
        """
        logger.info("🟡 启用本地 fallback（单版本修改规则）")
        mod_vers = {vid: txt for vid, txt in versions.items()}
        conflict_entities = []

        # 如果没有 shared_entities，直接返回
        if not shared_entities:
            return self._create_no_entity_response(versions)

        # 选取 1 ~ min(2, len(shared_entities)) 个实体用于本地扰动（防止过多）
        k = min(2, len(shared_entities))
        selected = random.sample(shared_entities, k)

        def perturb_entity(text: str, label: str, seed: int) -> str:
            # 简单扰动：年份 ±1、数字 ±1、机构替换为常见高校
            random.seed(seed)
            if label == "DATE" and re.search(r'\d{4}', text):
                try:
                    year = int(re.search(r'(\d{4})', text).group(1))
                    delta = random.choice([-1, 1])
                    new_year = year + delta
                    return text.replace(str(year), str(new_year))
                except:
                    pass
            if label in ("CARDINAL", "MONEY") and re.search(r'[\d,]+', text):
                m = re.search(r'([\d,]+)', text)
                if m:
                    raw_num = m.group(1)
                    try:
                        num = int(raw_num.replace(',', ''))
                        delta = random.choice([-1, 1])
                        new_num = num + delta
                        # 尝试保留千分位格式
                        new_num_str = f"{new_num:,}"
                        # 修复：确保替换回文本时不带逗号
                        return text.replace(raw_num, str(new_num))
                    except:
                        pass
            if label == "ORG" and ("大学" in text or "University" in text):
                candidates = ["北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学", "Harvard University", "Stanford University"]
                candidates = [c for c in candidates if c != text]
                if candidates:
                    return random.choice(candidates)
            if label == "PERSON":
                surnames = ["李", "王", "张", "刘", "陈"]
                english_surnames = ["Smith", "Jones", "Williams"]
                if re.search(r'[\u4e00-\u9fff]', text) and len(text) >= 2:
                    return random.choice(surnames) + text[1:]
                elif len(text.split()) > 1:
                    name_parts = text.split()
                    name_parts[-1] = random.choice(english_surnames)
                    return " ".join(name_parts)
            return text + "（变）"

        for idx, ent in enumerate(selected):
            text = ent["text"]
            label = ent["label"]
            vid = random.choice(list(mod_vers.keys()))
            new_text = perturb_entity(text, label, idx)
            if new_text == text:
                new_text = text + "（变）"
            if text in mod_vers[vid]:
                mod_vers[vid] = mod_vers[vid].replace(text, new_text, 1)
                conflict_entities.append({
                    "entity_text": text,
                    "entity_type": label,
                    "version_id": vid,
                    "original_text": text,
                    "modified_text": new_text,
                    "modification_type": "fallback_single_version_perturbation"
                })
            else:
                mod_vers[vid] = new_text + "。 " + mod_vers[vid]
                conflict_entities.append({
                    "entity_text": text,
                    "entity_type": label,
                    "version_id": vid,
                    "original_text": text,
                    "modified_text": new_text,
                    "modification_type": "fallback_insert_single_version"
                })

        return {
            "modified_versions": mod_vers,
            "conflict_entities": conflict_entities,
            "warning": "LLM 失败，启用本地 fallback 单版本冲突注入"
        }

# ---------- 后处理：校验并修复冲突 ----------
def post_validate_and_correct_conflicts(llm_result: Dict[str, Any], original_versions: Dict[str, str]) -> Dict[str, Any]:
    """
    确保：
    - 每个实体确实只在一个版本有修改（若LLM违反，尝试过滤/修正）
    - modified_text 出现在对应版本文本中（若不在，尝试替换 original_text -> modified_text 或插入）
    """
    mod_vers = llm_result.get("modified_versions", {}).copy()
    conflicts = llm_result.get("conflict_entities", [])
    valid_conflicts = []

    # 若 LLM 返回的是旧键名 conflict_entities 或 conflict_entities 为空，兼容处理
    if not conflicts:
        conflicts = llm_result.get("conflicts", [])

    for ent in conflicts:
        version_id = ent.get("version_id")
        modified_text = ent.get("modified_text")
        original_text = ent.get("original_text")
        entity_text = ent.get("entity_text")
        entity_type = ent.get("entity_type", ent.get("modification_type", "unknown"))
        if not version_id or not modified_text:
            continue

        # 若 modified_text 未出现在对应版本文本中，尝试替换 original_text -> modified_text
        if version_id in mod_vers and modified_text not in mod_vers[version_id]:
            if original_text and original_text in mod_vers[version_id]:
                mod_vers[version_id] = mod_vers[version_id].replace(original_text, modified_text, 1)
                logger.info(f"🔧 在版本 {version_id} 中替换原文以确保 modified_text 存在：{original_text} -> {modified_text}")
                valid_conflicts.append({
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "version_id": version_id,
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "modification_type": ent.get("modification_type", "unknown")
                })
            else:
                # 插入 modified_text 到版本开头
                mod_vers[version_id] = modified_text + "。 " + mod_vers[version_id]
                logger.info(f"🔧 在版本 {version_id} 开头插入 modified_text：{modified_text}")
                valid_conflicts.append({
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "version_id": version_id,
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "modification_type": ent.get("modification_type", "unknown")
                })
        else:
            valid_conflicts.append({
                "entity_text": entity_text,
                "entity_type": entity_type,
                "version_id": version_id,
                "original_text": original_text,
                "modified_text": modified_text,
                "modification_type": ent.get("modification_type", "unknown")
            })

    # 构建 version_conflicts 映射


    summary = {
        "total_modified_entities": len(valid_conflicts),
        "total_modifications": len(valid_conflicts),
    }

    return {
        "modified_versions": mod_vers,
        "conflict_details": valid_conflicts,
        "summary": summary
    }

# ---------- 冲突注入器 ----------
class ConflictInjector:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def inject_into_versions(
        self,
        versions: Dict[str, str],
        min_entities_to_modify: int = 1,
        max_entities_to_modify: int = 5
    ) -> Dict[str, Any]:
        logger.info("🔍 步骤1：提取跨版本共享实体...")
        shared_entities = find_shared_entities_across_versions(versions)
        if not shared_entities:
            logger.warning("⚠️ 无共享实体，跳过冲突注入")
            return self._empty_response(versions)

        logger.info("🤖 步骤2：调用 LLM 协同生成冲突（单版本修改）...")
        llm_result = self.llm.inject_cross_version_conflicts(
            versions=versions,
            shared_entities=shared_entities,
            min_entities_to_modify=min_entities_to_modify,
            max_entities_to_modify=max_entities_to_modify
        )

        logger.info("🔍 步骤3：后处理与校验...")
        processed = post_validate_and_correct_conflicts(llm_result, versions)

        # 构建最终精简输出
        conflicts = processed.get("conflict_details", [])
        output = {
            "modified_versions": processed["modified_versions"],
            "conflicts": [
                {
                    "entity_text": c["entity_text"],
                    "entity_type": c.get("entity_type", c.get("type", "unknown")),
                    "version_id": c["version_id"],
                    "original_text": c["original_text"],
                    "modified_text": c["modified_text"],
                    "modification_type": c.get("modification_type", "unknown")
                } for c in conflicts
            ],
            "version_conflicts": processed.get("version_conflicts", {}),
            "summary": processed.get("summary", {})
        }
        return output

    def _empty_response(self, versions):
        return {
            "modified_versions": {vid: txt for vid, txt in versions.items()},
            "conflicts": [],
            "version_conflicts": {},
            "summary": {"total_modified_entities": 0, "note": "no_shared_entities"}
        }

# ---------- I/O ----------
def load_rewritten_news(path: str) -> List[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ 加载失败: {e}")
        return []

    # 支持多种格式
    if isinstance(data, list) and data and "rewrite_versions" in data[0]:
        out = []
        for item in data:
            input_text = item.get("input_text", "") or item.get("original_text", "")
            versions_dict = {}
            for rw in item.get("rewrite_versions", []):
                vid = rw.get("version_id", "")
                text = rw.get("final_rewritten_text", "")
                if vid.startswith("rewritten_v") and text is not None:
                    versions_dict[vid] = text
            for i in range(1, 4):
                key = f"rewritten_v{i}"
                if key not in versions_dict:
                    versions_dict[key] = ""
            full_item = {"input_text": input_text, **versions_dict}
            out.append(full_item)
        return out
    elif isinstance(data, dict) and any(k.startswith("rewritten_v") for k in data.keys()):
        return [data]
    elif isinstance(data, dict) and "top_rewrites" in data:
        rewrites = data["top_rewrites"]
        item = {
            "rewritten_v1": rewrites[0].get("final_rewritten_text", "") if len(rewrites) > 0 else "",
            "rewritten_v2": rewrites[1].get("final_rewritten_text", "") if len(rewrites) > 1 else "",
            "rewritten_v3": rewrites[2].get("final_rewritten_text", "") if len(rewrites) > 2 else "",
            "input_text": data.get("input_text", "")
        }
        return [item]
    elif isinstance(data, list):
        return data
    else:
        logger.warning("⚠️ 未知输入格式，尝试解析顶层字段")
        return [data]

def save_output(data, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_incremental(data, path: str):
    """追加写入一行数据"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_processed_ids(path: str) -> set:
    """读取已生成的 JSONL 文件，获取已处理的 ID"""
    processed_ids = set()
    if not os.path.exists(path):
        return processed_ids
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj and obj["id"]:
                    processed_ids.add(str(obj["id"]))
            except:
                continue
    return processed_ids

def main():
    parser = argparse.ArgumentParser(description="✅ 跨版本实体冲突注入（支持断点续传）")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL (支持追加)")
    parser.add_argument("--min-entities", type=int, default=1)
    parser.add_argument("--max-entities", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # 1. 初始化配置
    try:
        config = get_model_deployment_config(args.model or DEFAULT_STUDENT_MODEL)
    except Exception:
        config = {"model": args.model or DEFAULT_STUDENT_MODEL}

    try:
        llm = LLMClient(model_config=config)
    except Exception as e:
        logger.exception(f"❌ 无法初始化 LLM: {e}")
        return

    injector = ConflictInjector(llm)

    # 2. 加载数据
    data_list = load_rewritten_news(args.input)
    if not data_list:
        logger.error("❌ 无输入数据")
        return

    # 3. 断点续传检查
    # 强制将输出后缀改为 .jsonl 以支持追加
    output_path = args.output if args.output.endswith('.jsonl') else args.output + ".jsonl"
    processed_ids = get_processed_ids(output_path)
    if processed_ids:
        logger.info(f"🔄 检测到断点：已跳过 {len(processed_ids)} 条已处理的数据")

    # 4. 循环处理
    for item in tqdm(data_list, desc="🔄 注入冲突"):
        current_id = str(item.get('id', ''))

        # 检查是否已处理过
        if current_id and current_id in processed_ids:
            continue

        versions = {}
        for i in range(1, 4):
            vid = f"rewritten_v{i}"
            versions[vid] = item.get(vid, "").strip()

        non_empty = [v for v in versions.values() if v]
        if len(non_empty) < 2:
            continue

        try:
            res = injector.inject_into_versions(
                versions,
                min_entities_to_modify=args.min_entities,
                max_entities_to_modify=args.max_entities
            )
            result_entry = {
                "input_text": item.get("input_text", ""),
                "id": item.get("id"),
                "injection_result": res
            }
            # --- 关键修改：实时保存 ---
            save_incremental(result_entry, output_path)

        except Exception as e:
            logger.exception(f"❌ 处理失败 (ID={current_id}): {e}")
            # 即使失败也记录一下，避免下次卡在这一条（或者你可以选不保存以便下次重试）
            # save_incremental({"id": current_id, "error": str(e)}, output_path)

    logger.info(f"✅ 完成！所有结果已保存至 (JSONL格式): {output_path}")
    logger.info(f"提示：如果需要标准JSON，可运行: cat {output_path} | jq -s . > final.json")

if __name__ == "__main__":
    main()
