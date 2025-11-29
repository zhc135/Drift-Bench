#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rewrite_agent_langgraph.py Strict implementation of rewrite agent based on Langgraph (requires langgraph installation).
- Uses StateGraph to register nodes and routing
- Records every tool/model call to state["tool_calls"]
- Node-level trace records to state["trace"]
- Evaluator prioritizes reduced_text as reference
- Iterative rewriting: each iteration directly modifies problematic parts based on previous version
- Evaluation logic modified: passes as long as faithfulness evaluation passes, no longer enforces diversity threshold
- Multi-API key support for faster parallel processing
"""
import json
import random
import re
import time
import uuid
import logging
import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypedDict
from types import SimpleNamespace

# Configuration import: can be adjusted based on project
try:
    from config_models import LOCAL_MODELS
except Exception:
    LOCAL_MODELS = {}

try:
    from config_runtime import DEBUG_MODE, REQUEST_TIMEOUT
except Exception:
    DEBUG_MODE = True
    REQUEST_TIMEOUT = 60

try:
    from config import MODEL_CONFIGS, get_model_deployment_config, DEFAULT_TEACHER_MODEL
except Exception:
    MODEL_CONFIGS = {}
    def get_model_deployment_config(model_name: str) -> Dict[str, Any]:
        return MODEL_CONFIGS.get(model_name, {})
    DEFAULT_TEACHER_MODEL = "gpt-4o"

# Force dependency on langgraph
try:
    from langgraph.graph import StateGraph  # type: ignore
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

# langchain openai wrapper (based on environment)
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
        def __repr__(self):
            return f"SystemMessage({self.content[:30]})"
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
        def __repr__(self):
            return f"HumanMessage({self.content[:30]})"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rewrite_agent_langgraph")

# ---------------- Types ----------------
class AgentState(TypedDict, total=False):
    news_text: str
    original_text: str
    num_versions: int
    rewrite_attempts: int
    max_attempts: int
    quality_threshold: float
    model_config: Dict[str, Any]
    segments: List[Dict[str, Any]]
    analyzed_segments: List[Dict[str, Any]]
    masks: List[Any]
    reduced_versions: List[Dict[str, Any]]
    rewritten_versions: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    best_score: float
    best_version: Dict[str, Any]
    trace: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    errors: List[str]
    terminated_at: str
    final_status: str
    summary: Dict[str, Any]

# ---------------- Helper Functions ----------------
def _parse_json_from_model_output(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        try:
            cleaned = match.group().replace("'", '"')
            return json.loads(cleaned)
        except Exception:
            return None

def _create_llm(model_config: Optional[Dict[str, Any]] = None, api_config: Optional[Dict[str, Any]] = None):
    if model_config is None:
        model_config = get_model_deployment_config(DEFAULT_TEACHER_MODEL)
    temperature = model_config.get("temperature", 0.7)
    max_tokens = model_config.get("max_tokens", 2000)
    model_name = model_config.get("model_name", model_config.get("model", DEFAULT_TEACHER_MODEL))
    
    # Use provided API config or get from model_config
    if api_config:
        api_key = api_config.get("api_key", "")
        base_url = api_config.get("base_url", "https://api.openai.com/v1")
    else:
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
    
    if api_key == "EMPTY":
        api_key = "vllm"
    if DEBUG_MODE:
        logger.info(f"Initializing local vLLM: model={model_name}")
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI client unavailable, please install or adapt your LLM client.")
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=base_url,
        request_timeout=REQUEST_TIMEOUT
    )
    return llm

# Mapping helper: map model output keep_id (various formats) to full_tag_id
def map_keep_ids_to_full(segment_id: str, standardized_tags: List[Dict[str, Any]], keep_raw: List[str]):
    mapped = []
    warnings = []
    tag_by_local = {t["tag_id"]: t["full_tag_id"] for t in standardized_tags if t.get("tag_id")}
    full_ids = {t["full_tag_id"] for t in standardized_tags}
    for kid in (keep_raw or []):
        if not isinstance(kid, str):
            continue
        kid = kid.strip()
        if not kid:
            continue
        # already full form?
        if kid.startswith(segment_id + "_"):
            if kid in full_ids:
                mapped.append(kid)
            else:
                mapped.append(kid)
                warnings.append(f"full_id_format_not_found:{kid}")
            continue
        # exact local tag_id match
        if kid in tag_by_local:
            mapped.append(tag_by_local[kid])
            continue
        # exact match to any full id
        if kid in full_ids:
            mapped.append(kid)
            continue
        # numeric heuristic
        d = re.search(r'\d+', kid)
        if d:
            num = d.group(0)
            found = False
            for cand_local, cand_full in tag_by_local.items():
                if num in cand_local or num in cand_full:
                    mapped.append(cand_full)
                    found = True
                    break
            if found:
                continue
            cand = f"{segment_id}_local_{num}"
            if cand in full_ids:
                mapped.append(cand)
                continue
        # text match against tag text/sentence
        text_match = next((t["full_tag_id"] for t in standardized_tags if (t.get("text") or "").strip() == kid or (t.get("sentence") or "").strip() == kid), None)
        if text_match:
            mapped.append(text_match)
            continue
        # fallback
        fallback = f"{segment_id}_{kid}"
        mapped.append(fallback)
        warnings.append(f"fallback_mapped:{kid}->{fallback}")
    deduped = list(dict.fromkeys(mapped))
    return deduped, warnings

# ---------------- Tools ----------------
def create_segmenter_tool():
    def segmenter(input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get("text", "") if isinstance(input_data, dict) else ""
        try:
            if not isinstance(text, str):
                raise ValueError("Parameter 'text' must be a string")
            pattern = re.compile(r'(?s).*?(?:\n\s*\n|$)')
            segments = []
            for idx, m in enumerate(pattern.finditer(text), start=1):
                raw = m.group()
                trimmed = raw.strip()
                if not trimmed:
                    continue
                local_off = raw.find(trimmed)
                start_char = m.start() + local_off
                end_char = start_char + len(trimmed)
                segments.append({
                    "segment_id": f"seg_{idx:03d}",
                    "text": trimmed,
                    "start_char": start_char,
                    "end_char": end_char
                })
            return {"segments": segments}
        except Exception as e:
            return {"error": str(e)}
    return SimpleNamespace(name="segmenter", func=segmenter)

def create_analyze_segment_tool(model_config: Dict[str, Any], max_retries: int = 2):
    def analyze_segment(input_data: Dict[str, Any]) -> Dict[str, Any]:
        segment_id = input_data.get("segment_id", "seg_unknown")
        text = input_data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return {"segment_id": segment_id, "tags": [], "reduced_versions": [], "tagged_text": "", "raw_model_output": ""}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        system_prompt = (
            "You are a professional news content analyst. Task: 1. Analyze the following news paragraph, identify key information and label with tags. 2. Analyze in detail which information in the original text can be independently rewritten to obtain multiple new articles related to this topic, and fill the corresponding tags into reduced_versions.\n\n"
            " Strictly output JSON, strictly do not output any additional text or explanation.\n\n"
            "Output structure requirements (must be followed):\n"
            "- tags: list, each element contains fields {\"tag_id\": \"local_id\", \"tag_type\": ..., \"text\": ..., \"sentence\": \"...\", \"reason\": \"...\"}\n"
            "- reduced_versions: list, containing several versions (e.g., reduced_v1, reduced_v2, reduced_v3), each version must contain:\n"
            " {\"version_id\":\"reduced_v1\", \"keep_tag_ids\": [\"{segment_id}_{tag_id}\" or \"tag_id\"], \"suggestions\": \"Rewriting direction and suggested short sentences\", \"reason\":\"Reason for selecting these contents in this version\"}\n"
            "Explanation: Model can output keep_tag_ids in two forms:\n"
            " 1) Already in full format '{segment_id}_{tag_id}' (recommended);\n"
            " 2) Can also output only local tag_id (e.g., 'tag1'), backend will attempt to map to full_tag_id ({segment_id}_tag1).\n"
            "Important: keep_tag_ids represent \"tags that should be kept in this version\". Tags not included in keep_tag_ids will be considered as \"removal/mask\" targets (i.e., will be removed later).\n\n"
            "Other requirements:\n"
            "- Each tag's sentence field must be a continuous segment of text from the original (do not concatenate).\n"
            "- Do not output any non-JSON text.\n"
        )
        human_prompt = text
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed is None:
                    last_err = f"Cannot parse JSON (attempt {attempt+1}): raw={(raw or '')[:400]}"
                    time.sleep(0.3 * (attempt + 1))
                    continue
                # Standardize tags: supplement full_tag_id and safe_tag
                tags = parsed.get("tags", []) or []
                standardized_tags = []
                for tag in tags:
                    local_tag_id = str(tag.get("tag_id") or "").strip()
                    if not local_tag_id:
                        local_tag_id = f"tag_{uuid.uuid4().hex[:8]}"
                    full_tag_id = f"{segment_id}_{local_tag_id}"
                    safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', full_tag_id)
                    new_tag = dict(tag)
                    new_tag["tag_id"] = local_tag_id
                    new_tag["full_tag_id"] = full_tag_id
                    new_tag["safe_tag"] = safe_tag
                    standardized_tags.append(new_tag)
                # Parse reduced_versions (map keep_tag_ids to full_tag_id)
                reduced_versions_raw = parsed.get("reduced_versions", []) or parsed.get("reduced_masks", []) or []
                reduced_versions = []
                for idx, item in enumerate(reduced_versions_raw):
                    if not isinstance(item, dict):
                        continue
                    if "version_id" in item:
                        vid = item.get("version_id")
                        keep_raw = item.get("keep_tag_ids", []) or item.get("keep_ids", []) or []
                        sugg = item.get("suggestions", "") or item.get("advice", "")
                        reason = item.get("reason", "")
                    else:
                        keys = list(item.keys())
                        if keys:
                            vid = keys[0]
                            sub = item.get(vid, {}) or {}
                            keep_raw = sub.get("keep_tag_ids") or sub.get("keep_ids") or sub.get("keep") or sub.get("mask_ids") or []
                            sugg = sub.get("suggestions", "") or sub.get("advice", "")
                            reason = sub.get("reason", "")
                        else:
                            continue
                    mapped_keep = []
                    for kid in (keep_raw or []):
                        if not isinstance(kid, str):
                            continue
                        kid = kid.strip()
                        if kid.startswith(segment_id + "_"):
                            mapped_keep.append(kid)
                        else:
                            match = next((t["full_tag_id"] for t in standardized_tags if t.get("tag_id") == kid or t.get("full_tag_id") == kid), None)
                            if match:
                                mapped_keep.append(match)
                            else:
                                mapped_keep.append(f"{segment_id}_{kid}")
                    reduced_versions.append({
                        "version_id": vid if isinstance(vid, str) else f"reduced_v{idx+1}",
                        "keep_tag_ids": list(dict.fromkeys(mapped_keep)),
                        "suggestions": sugg,
                        "reason": reason
                    })
                # Generate tagged_text (wrap sentence with safe_tag)
                sentences_to_tag = []
                seen = set()
                for tag in standardized_tags:
                    sent = (tag.get("sentence") or "").strip()
                    if sent and sent not in seen:
                        sentences_to_tag.append(sent)
                        seen.add(sent)
                tagged_text = text
                for sent in reversed(sentences_to_tag):
                    tag_candidates = [t.get("safe_tag") for t in standardized_tags if (t.get("sentence") or "").strip() == sent]
                    tag_name = tag_candidates[0] if tag_candidates else ("tag_" + uuid.uuid4().hex[:6])
                    tagged_text = tagged_text.replace(sent, f"<{tag_name}>{sent}</{tag_name}>", 1)
                # Automatically generate auto tag for paragraphs not covered by any tag
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
                auto_count = 0
                for para in paragraphs:
                    covered = False
                    for sent in sentences_to_tag:
                        if sent and sent in para:
                            covered = True
                            break
                    if not covered:
                        auto_count += 1
                        auto_local = f"auto_{auto_count}"
                        full_tag_id = f"{segment_id}_{auto_local}"
                        safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', full_tag_id)
                        auto_tag = {
                            "tag_id": auto_local,
                            "full_tag_id": full_tag_id,
                            "safe_tag": safe_tag,
                            "tag_type": "background_auto",
                            "text": para,
                            "sentence": para,
                            "reason": "auto-generated to preserve untagged paragraph"
                        }
                        standardized_tags.append(auto_tag)
                        tagged_text = tagged_text.replace(para, f"<{safe_tag}>{para}</{safe_tag}>", 1)
                parsed_out = {
                    "segment_id": segment_id,
                    "tags": standardized_tags,
                    "reduced_versions": reduced_versions,
                    "tagged_text": tagged_text,
                    "raw_model_output": raw,
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt
                }
                return parsed_out
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"segment_id": segment_id, "tags": [], "reduced_versions": [], "tagged_text": "", "raw_model_output": "", "error": last_err}
    return SimpleNamespace(name="analyze_segment", func=analyze_segment)

def create_apply_reduction_mask_tool():
    def apply_reduction_mask(input_data: Dict[str, Any]) -> Dict[str, Any]:
        segments = input_data.get("segments", [])
        mask = set(input_data.get("mask", []))
        reduced_parts = []
        for seg in segments:
            tagged_text = seg.get("tagged_text", seg.get("text", ""))
            seg_id = seg.get("segment_id", "")
            tags_to_remove = set()
            for t in seg.get("tags", []):
                fid = t.get("full_tag_id")
                if fid and fid in mask:
                    st = t.get("safe_tag") or re.sub(r'[^a-zA-Z0-9_]', '_', fid)
                    tags_to_remove.add(st)
            current = tagged_text
            for tag_name in tags_to_remove:
                current = re.sub(rf"<{re.escape(tag_name)}>[\s\S]*?</{re.escape(tag_name)}>", "", current)
            current = re.sub(r'<[^>]+>', '', current)
            cleaned = re.sub(r'\n\s*\n', '\n\n', current).strip()
            reduced_parts.append({"segment_id": seg_id, "text": cleaned})
        combined = "\n\n".join(p["text"] for p in reduced_parts)
        return {"reduced_text": combined, "reduced_segments": reduced_parts}
    return SimpleNamespace(name="apply_reduction_mask", func=apply_reduction_mask)

# Modified create_rewrite_tool: supports iterative modification mode
def create_rewrite_tool(model_config: Dict[str, Any], max_retries: int = 2):
    def _safe_str(x):
        try:
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    def rewrite_text(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supports iterative modification mode: based on previous_text and evaluation_feedback for partial modifications
        Return fields: rewritten_text, raw_model_output, system_prompt, human_prompt, used_feedback, evaluation_summary, sampling_params
        """
        import json as _json
        import re as _re
        import time as _time

        # External injectable dependencies (use global if defined in global scope)
        _llm_creator = globals().get("_create_llm", _create_llm)
        _SystemMessage = globals().get("SystemMessage", SystemMessage)
        _HumanMessage = globals().get("HumanMessage", HumanMessage)
        _global_safe_str = globals().get("_safe_str", _safe_str)

        # Read parameters from input_data, provide reasonable defaults
        reduced_text = input_data.get("reduced_text", None)
        if reduced_text is None:
            reduced_text = input_data.get("text", "")
        previous_text = input_data.get("previous_text", None)
        evaluation_summary = input_data.get("evaluation_summary", None)
        preserve_tags = input_data.get("preserve_tags", []) or []
        feedback = input_data.get("feedback", None)
        sampling_params = input_data.get("sampling_params", {}) or {}

        # Optional configuration items
        local_max_retries = input_data.get("max_retries", max_retries)
        local_model_config = input_data.get("model_config", model_config or {})
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None

        # Determine base text: if previous_text exists, modify based on it; otherwise generate based on reduced_text
        if previous_text and isinstance(previous_text, str) and previous_text.strip():
            base_text = previous_text.strip()
            is_iterative = True
        else:
            base_text = reduced_text.strip() if isinstance(reduced_text, str) else ""
            is_iterative = False

        if not base_text:
            return {"rewritten_text": "", "error": "Base text is empty"}

        # Build prompt
        if is_iterative:
            system_prompt = (
                "You are a professional news editor responsible for making precise modifications to text based on evaluation feedback.\n\n"
                "**Modification Requirements:**\n"
                "1. Only modify the problematic parts pointed out in the evaluation feedback, keep other content unchanged\n"
                "2. For \"missing\" content: Add missing information at appropriate locations\n"
                "3. For \"extra\" content: Delete or correct extra information\n"
                "4. For \"suggested modifications\": Make precise modifications according to suggestions\n"
                "5. Maintain text fluency and coherence\n\n"
                "Please strictly modify based on the previous version, do not rewrite the entire text."
            )
            human_parts = []
            human_parts.append("Previous version:\n" + base_text)

            if evaluation_summary and isinstance(evaluation_summary, dict):
                faith = evaluation_summary.get("faithfulness", {}) or {}
                if faith.get("missing"):
                    human_parts.append("\nMissing content (needs to be added):")
                    for m in faith.get("missing", []):
                        if isinstance(m, dict):
                            human_parts.append(f"- {m.get('text', '')} (location hint: {m.get('where_hint', '')})")
                        else:
                            human_parts.append(f"- {str(m)}")
                if faith.get("extra"):
                    human_parts.append("\nExtra content (needs to be deleted or corrected):")
                    for ex in faith.get("extra", []):
                        if isinstance(ex, dict):
                            human_parts.append(f"- {ex.get('text', '')} (location hint: {ex.get('where_hint', '')})")
                        else:
                            human_parts.append(f"- {str(ex)}")
                if faith.get("suggested_edits"):
                    human_parts.append("\nSuggested edits:")
                    for edit in faith.get("suggested_edits", []):
                        if isinstance(edit, dict):
                            action = edit.get('action', 'Modify')
                            text = edit.get('text', '')
                            hint = edit.get('position_hint', '')
                            human_parts.append(f"- {action}: {text} (position: {hint})")
                        else:
                            human_parts.append(f"- {str(edit)}")

            if feedback:
                if isinstance(feedback, str):
                    human_parts.append(f"\nAdditional feedback: {feedback}")
                else:
                    try:
                        human_parts.append("\nAdditional feedback: " + _json.dumps(feedback, ensure_ascii=False))
                    except Exception:
                        human_parts.append("\nAdditional feedback: " + str(feedback))

            human_parts.append("\nPlease output the modified complete text (only modify problematic parts, keep other content unchanged):")
            human_prompt = "\n".join(human_parts)
        else:
            reduced_text_for_prompt = reduced_text if isinstance(reduced_text, str) else ""
            system_prompt = (
                "You are a professional news editor specializing in paragraph-by-paragraph optimization of news manuscripts. Your core task is to precisely rewrite the single news paragraph I provide, ensuring style transformation and language innovation while strictly maintaining original meaning, facts, and length, so it can be directly replaced back into the original text.\n\n"
                "I will provide the following information in order, please pay close attention:\n\n"
                "[Full Text]: The content of the entire article for your analysis.\n"
                "[Paragraph to Rewrite]: The current paragraph you need to process.\n\n"
                "Core Rewriting Principles (Must Strictly Follow)\n\n"
                "Zero Fact Alteration: Strictly prohibit adding, deleting, or distorting any core facts, data, or character viewpoints.\n\n"
                "Length Consistency: The word count, information density, and length of the rewritten paragraph should be highly similar to the original, ensuring it can be directly \"embedded\" into the original position.\n\n"
                "Logical Cohesion: The beginning and end of the rewritten paragraph must smoothly transition with the preceding and following paragraphs, maintaining the overall fluency of the article.\n\n"
                "Full Text: {reduced_text}\n\n"
                "Paragraph to Rewrite: {base_text}\n\n"
            ).format(reduced_text=reduced_text_for_prompt, base_text=base_text)

            human_parts = []
            human_parts.append("Writing Goal: Rewrite this paragraph based on the provided full text content and this paragraph's content")
            human_parts.append("\nReference Text:\n" + base_text)
            if feedback:
                if isinstance(feedback, str):
                    human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + feedback)
                else:
                    try:
                        human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + _json.dumps(feedback, ensure_ascii=False))
                    except Exception:
                        human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + str(feedback))
            human_parts.append("\nOnly return the rewritten paragraph text (no extra explanations).")
            human_prompt = "\n\n".join(human_parts)

        last_err = None
        for attempt in range(1, local_max_retries + 1):
            try:
                # Merge model configuration with sampling parameters
                local_model_cfg = dict(local_model_config or {})
                for k, v in (sampling_params or {}).items():
                    if v is not None:
                        local_model_cfg[k] = v

                if not _llm_creator:
                    return {"rewritten_text": "", "error": "_create_llm not defined"}

                llm = _llm_creator(local_model_cfg, api_config)
                messages = [_SystemMessage(content=system_prompt), _HumanMessage(content=human_prompt)]

                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)

                rewritten = (raw or "").strip()
                rewritten = _re.sub(r'<[^>]+>', '', rewritten).strip()

                if not rewritten:
                    last_err = f"Empty output (attempt {attempt})"
                    _time.sleep(0.2 * attempt)
                    continue

                result = {
                    "rewritten_text": rewritten,
                    "raw_model_output": raw,
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt,
                    "used_feedback": _global_safe_str(feedback or evaluation_summary),
                    "evaluation_summary": evaluation_summary,
                    "sampling_params": sampling_params,
                    "is_iterative": is_iterative,
                    "preserve_tags": preserve_tags,
                }
                return result

            except Exception as e:
                last_err = str(e)
                _time.sleep(0.2 * attempt)

        return {"rewritten_text": "", "error": last_err}

    return SimpleNamespace(name="rewrite_text", func=rewrite_text)

def create_faithfulness_evaluator(model_config: Dict[str, Any], max_retries: int = 2):
    def evaluate(input_data: Dict[str, Any]) -> Dict[str, Any]:
        reduced = input_data.get("reduced_text", None)
        if reduced is None:
            reduced = input_data.get("original_text", "").strip()
        else:
            reduced = str(reduced).strip()
        rewritten = input_data.get("rewritten_text", "").strip()
        if not reduced:
            return {"passed": False, "score": 0.0, "feedback": "Reference text (reduced) is empty"}
        if not rewritten:
            return {"passed": False, "score": 0.0, "feedback": "Rewritten text is empty"}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        # Require more detailed structured output: missing/extra/suggested_edits
        prompt = f"""You are a fact-checking expert. Please strictly judge whether the rewritten text is faithful to the given reference text (the reference text is the reduced version).
Reference (Reduced) Text: {reduced}
Rewritten Text: {rewritten}
Please output strict JSON, containing fields:
- passed: true/false
- score: 0.0~1.0
- missing: list, containing objects {{ "text": "...", "where_hint": "..." }} (indicating which facts or key points from the reference are missing in the rewrite)
- extra: list, containing objects {{ "text": "...", "where_hint": "..." }} (indicating facts added in the rewrite that should not be there or are uncertain)
- suggested_edits: list, containing objects {{ "action":"insert|delete|rewrite", "text":"...", "position_hint":"..." }}
- explanation: natural language explanation of the issues
Only return JSON, no additional text or explanation. Example: {{\"passed\":true,\"score\":0.95, ...}}"""
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content="You are a strict fact-checker. Only output JSON."), HumanMessage(content=prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed and "passed" in parsed:
                    parsed["raw_model_output"] = raw
                    parsed["system_prompt"] = "You are a strict fact-checker. Only output JSON."
                    parsed["human_prompt"] = prompt
                    # Compatibility: ensure suggested_edits is in list format
                    if "suggested_edits" not in parsed:
                        parsed.setdefault("suggested_edits", [])
                    return parsed
                last_err = "Cannot parse returned JSON"
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"passed": False, "score": 0.0, "feedback": last_err or "Evaluation failed", "suggested_edits": []}
    return SimpleNamespace(name="faithfulness_eval", func=evaluate)

def create_diversity_evaluator(model_config: Dict[str, Any], max_retries: int = 2):
    def evaluate(input_data: Dict[str, Any]) -> Dict[str, Any]:
        reduced = input_data.get("reduced_text", None)
        if reduced is None:
            reduced = input_data.get("original_text", "").strip()
        else:
            reduced = str(reduced).strip()
        rewritten = input_data.get("rewritten_text", "").strip()
        if not reduced or not rewritten:
            return {"passed": False, "score": 0.0, "feedback": "Reference or rewritten text is empty"}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        prompt = f"""You are a text diversity evaluation expert. Judge whether the rewritten text is significantly different in expression from the reference (reduced) text.
Reference (Reduced) Text: {reduced}
Rewritten Text: {rewritten}
Please answer (only output JSON):
- Diversity score (0.0~1.0, 0=almost identical, 1=highly different expression but semantically consistent)
- Is sufficiently diverse (true/false)
- Improvement suggestions (try diversity from style, role, dissemination form perspectives)
Output strictly as JSON, e.g.: {{\"passed\":true,\"score\":..,\"feedback\":\"...\"}}"""
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content="You are a text diversity expert. Only output JSON."), HumanMessage(content=prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed and "passed" in parsed:
                    parsed["raw_model_output"] = raw
                    parsed["system_prompt"] = "You are a text diversity expert. Only output JSON."
                    parsed["human_prompt"] = prompt
                    return parsed
                last_err = "Cannot parse returned JSON"
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"passed": False, "score": 0.0, "feedback": last_err or "Evaluation failed"}
    return SimpleNamespace(name="diversity_eval", func=evaluate)

# ---------------- build tools ----------------
def build_tools_for_agent(model_config: Dict[str, Any]) -> Dict[str, Callable]:
    tools = [
        create_segmenter_tool(),
        create_analyze_segment_tool(model_config),
        create_apply_reduction_mask_tool(),
        create_rewrite_tool(model_config),
        create_faithfulness_evaluator(model_config),
        create_diversity_evaluator(model_config),
    ]
    return {t.name: t.func for t in tools}

# ---------------- Graph nodes (stateless functions that operate on state) ----------------
def make_graph_nodes(tools: Dict[str, Callable], max_analyze_workers: int = 4):
    # common helper to record tool calls
    def record_tool_call(state: Dict[str, Any], node: str, tool: str, inp: Dict[str, Any], out: Dict[str, Any], duration: Optional[float] = None):
        state.setdefault("tool_calls", [])
        entry = {
            "ts": time.time(),
            "node": node,
            "tool": tool,
            "input": inp,
            "output": out
        }
        if duration is not None:
            entry["duration"] = duration
        state["tool_calls"].append(entry)
        if DEBUG_MODE:
            logger.debug("tool_call: %s %s duration=%.3f", node, tool, duration or 0.0)
        return state

    def node_segmenter(state: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "segmenter_start", "ts": start_ts})
        txt = state.get("news_text", "")
        t0 = time.time()
        res = tools["segmenter"]({"text": txt})
        t1 = time.time()
        duration = t1 - t0
        state = record_tool_call(state, "segmenter", "segmenter", {"text": txt}, res, duration=duration)
        if "error" in res:
            state.setdefault("errors", []).append(f"segmenter_error:{res['error']}")
            state["segments"] = []
        else:
            state["segments"] = res["segments"]
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "segmenter_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    def node_analyze_segments(state: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "analyze_segments_start", "ts": start_ts})
        segments = state.get("segments", [])
        state.setdefault("tool_calls", [])
        analyzed_results = [None] * len(segments)
        def call_and_record(idx, seg):
            inp = {"segment_id": seg.get("segment_id"), "text": seg.get("text")}
            t0 = time.time()
            out = tools["analyze_segment"](inp)
            t1 = time.time()
            out.setdefault("segment_id", inp["segment_id"])
            out.setdefault("text", inp["text"])
            out.setdefault("tags", out.get("tags", []))
            duration = t1 - t0
            record_tool_call(state, "analyze_segments", "analyze_segment", inp, out, duration=duration)
            return idx, out
        workers = min(max_analyze_workers, max(1, len(segments)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(call_and_record, idx, seg): idx for idx, seg in enumerate(segments)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, out = fut.result()
                except Exception as e:
                    out = {"segment_id": segments[idx].get("segment_id"), "text": segments[idx].get("text"), "tags": [], "error": str(e)}
                record_tool_call(state, "analyze_segments", "analyze_segment", {"segment_id": segments[idx].get("segment_id"), "text": segments[idx].get("text")}, out)
                analyzed_results[idx] = out
        state["analyzed_segments"] = analyzed_results
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "analyze_segments_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    def node_generate_masks(state: Dict[str, Any]) -> Dict[str, Any]:
        """ Prioritize using reduced_versions returned by analyze_segment (version suggestions for each segment) to construct global keep_tag_lists (merge keep for each segment by version).
        If analyze does not return reasonable keep lists (all versions empty or abnormal), fallback to simple random mask generation.
        Final output:
        state["masks"] : list of masks (each is a list of full_tag_id to remove)
        state["mask_keep_lists"] : list of keep full_tag_id lists (per version)
        state["mask_suggestions"] : list of dict(s) containing version_id / suggestions / keep_tag_ids
        """
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "generate_masks_start", "ts": start_ts})
        segs = state.get("analyzed_segments", []) or []
        num_versions = int(state.get("num_versions", 3))
        # gather all full tag ids
        all_full_ids = []
        for seg in segs:
            for t in seg.get("tags", []):
                fid = t.get("full_tag_id")
                if fid:
                    all_full_ids.append(fid)
        # collect per-segment reduced_versions
        per_seg_rvs = []
        version_ids = []
        for seg in segs:
            rvs = seg.get("reduced_versions", []) or []
            per_seg_rvs.append((seg.get("segment_id"), rvs))
            if rvs and not version_ids:
                version_ids = [rv.get("version_id") for rv in rvs if rv.get("version_id")]
        if not version_ids:
            version_ids = [f"reduced_v{i+1}" for i in range(num_versions)]
        mapped_keep_lists = []
        mask_suggestions = []
        # For each version id, aggregate keep_tag_ids and suggestions across segments
        for vid in version_ids:
            keep_union = set()
            suggestions = []
            for seg_id, rvs in per_seg_rvs:
                for rv in rvs:
                    if rv.get("version_id") == vid:
                        keep_list = rv.get("keep_tag_ids", []) or []
                        for k in keep_list:
                            if isinstance(k, str) and k:
                                keep_union.add(k)
                        s = rv.get("suggestions")
                        if s:
                            suggestions.append(s)
            mapped_keep_lists.append(list(keep_union))
            mask_suggestions.append({"version_id": vid, "suggestions": suggestions, "keep_tag_ids": list(keep_union)})
        # If no keep information f
