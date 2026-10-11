# #!/usr/bin/env python3
# import subprocess
# import sys
# import os
# import re
# import json
# from threading import Lock
# import time
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple
# from langchain_community.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage
# import logging
# from collections import Counter, defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import argparse
# import numpy as np
# import jieba
# import random
# # 导入配置
# from config import (
#     get_model_deployment_config,
#     DEFAULT_TEACHER_MODEL,
#     MAX_WORKERS,
#     REQUEST_TIMEOUT,
#     LOG_LEVEL,
#     DEBUG_MODE,
#     ENABLE_PROGRESS_BAR,
#     API_RETRY_LIMIT,
#     COT_CONFIDENCE_THRESHOLD,
#     LOW_CONFIDENCE_THRESHOLD,
#     BATCH_SIZE,
# )

# # ==================== 设置随机种子以确保可复现性 ====================
# RANDOM_SEED = 42  # 固定随机种子

# def set_random_seeds(seed: int = RANDOM_SEED):
#     """设置所有随机种子以确保可复现性"""
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     # 设置TensorFlow和PyTorch的随机种子（如果可用）
#     try:
#         import torch
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#     except ImportError:
#         pass
#     try:
#         import tensorflow as tf
#         tf.random.set_seed(seed)
#     except ImportError:
#         pass

# # 设置随机种子
# set_random_seeds(RANDOM_SEED)

# # 日志配置
# logging.basicConfig(level=getattr(logging, LOG_LEVEL))
# logger = logging.getLogger(__name__)

# # 修复jieba缓存目录问题
# jieba_cache_dir = os.path.join(os.path.expanduser("~"), ".jieba_cache")
# os.makedirs(jieba_cache_dir, exist_ok=True)
# jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
# if DEBUG_MODE:
#     logger.info(f"🔧 设置jieba缓存目录: {jieba_cache_dir}")

# # ==================== 模型使用统计类 ====================
# class ModelUsageStats:
#     def __init__(self):
#         self.strict_model_name = ""
#         self.cot_model_name = ""
#         self.strict_model_calls = 0
#         self.cot_model_calls = 0
#         self.lock = Lock()

#     def increment_strict(self):
#         with self.lock:
#             self.strict_model_calls += 1

#     def increment_cot(self):
#         with self.lock:
#             self.cot_model_calls += 1

#     def get_stats(self) -> Dict[str, Any]:
#         with self.lock:
#             return {
#                 "timestamp": datetime.now().isoformat(),
#                 "strict_model": self.strict_model_name,
#                 "cot_model": self.cot_model_name,
#                 "strict_model_calls": self.strict_model_calls,
#                 "cot_model_calls": self.cot_model_calls,
#                 "total_calls": self.strict_model_calls + self.cot_model_calls,
#                 "random_seed": RANDOM_SEED  # 记录使用的随机种子
#             }

#     def print_summary(self):
#         stats = self.get_stats()
#         print("\n" + "="*60)
#         print("📊 模型使用统计摘要")
#         print("="*60)
#         print(f"📅 时间: {stats['timestamp']}")
#         print(f"🤖 严格模型: {stats['strict_model']}")
#         print(f"🤖 CoT模型: {stats['cot_model']}")
#         print(f"📞 严格模型调用次数: {stats['strict_model_calls']}")
#         print(f"📞 CoT模型调用次数: {stats['cot_model_calls']}")
#         print(f"📞 总调用次数: {stats['total_calls']}")
#         print(f"🎲 随机种子: {stats['random_seed']}")
#         print("="*60)

# # ==================== 语言质量问题定义 ====================
# LANGUAGE_QUALITY_PROBLEMS = {
#     "Language Quality and Style Consistency": {
#         "description": "Evaluate the language quality, style consistency, fluency, and overall readability of the fused text. Check for issues such as stylistic inconsistency, awkward expression, redundancy, inconsistent terminology, tone mismatch, sentence structure problems, or other language quality issues.",
#         "evaluation_method": "Analyze the language quality of the fused text from multiple dimensions: style consistency, fluency, terminology usage, sentence structure, and overall readability."
#     }
# }

# # 定义语言问题分类（已移除 cohesion_issues）
# LANGUAGE_PROBLEM_CATEGORIES = {
#     "stylistic_inconsistency": "风格不一致",
#     "awkward_expression": "表达生硬或不自然",
#     "redundant_repetition": "冗余重复",
#     "inconsistent_terminology": "术语不一致",
#     "tone_mismatch": "语气/语调不匹配",
#     "sentence_structure_problems": "句子结构问题",
#     "other_language_issues": "其他语言问题"
# }

# # ==================== 语言质量评估Prompt模板 (重新修改) ====================
# LANGUAGE_QUALITY_EVALUATE_PROMPT = """
# As a professional news editor and language expert, please evaluate the language quality and style consistency of the fused text by COMPARING it with the original text.

# ## Problem Description: {problem_description}

# ## CRITICAL EVALUATION RULES:
# 1. **COMPARE WITH ORIGINAL TEXT**: You must ALWAYS compare the fused text with the original text to identify language quality issues. Look for problems that appear in the fused text but not in the original, or problems that are worse in the fused text.
# 2. **FOCUS ON THE MOST SEVERE PROBLEM AT EACH TEXT LOCATION**: For each problematic text location, identify and report ONLY the single most significant language quality problem at that location. Different locations can have different primary problems.
# 3. **DO NOT LIMIT TO ONE PROBLEM PER ARTICLE**: Report ALL distinct problematic locations in the text, each with its own most severe problem. A single article may have multiple language issues at different locations.
# 4. **Severity Hierarchy (applied per location)**: For each problematic location, prioritize problems as follows (highest to lowest): Sentence Structure Problems, Awkward Expression, Stylistic Inconsistency, Tone Mismatch, Inconsistent Terminology, Redundant Repetition, Other Language Issues.
# 5. **Ignore Minor Issues**: If the most significant issue at a location is minor and does not substantially impact readability or professionalism, you may choose not to report that location.
# 6. **Exclude Factual/Literal Issues**: Do NOT include factual omission, logical sequence issues, or factual accuracy as language quality problems.

# ## EVALUATION REQUIREMENTS:
# 1. **Compare Thoroughly**: Examine both original and fused texts for Style Consistency, Expression Quality, Terminology Use, Sentence Structure, and Readability.
# 2. **Identify Changes That Degrade Quality**: Focus on changes from original to fused text that introduce or worsen language problems.
# 3. **Identify All Problematic Locations**: Find ALL text locations in the fused text where language problems exist (compared to original).
# 4. **Select Primary Issue Per Location**: For EACH problematic location, determine the single most impactful language problem based on the hierarchy above.
# 5. **Classify Each Issue**: Assign each identified primary issue to its corresponding category.
# 6. **Judge Severity Per Issue**: Assess if each issue is severe enough to warrant reporting.

# ## Text Content:
# **Original Text**: {original_text}
# **Fused Text**: {fused_text}

# ## Output Requirements:
# - Answer in the same language as the input text
# - Provide structured evidence in JSON format ALWAYS, regardless of whether any problems exist

# ## Output Format: Please return the evaluation in JSON format, including the following fields:
# - "problem_exists": Boolean, indicating whether ANY SIGNIFICANT language quality problems exist in the fused text (compared to original).
# - "evidence": Structured analysis with the following format:
#   - "analysis_summary": Brief summary of the overall language quality assessment, focusing on changes from original to fused.
#   - "identified_problems": A LIST of objects, each representing a distinct problematic location in the fused text. Each object should contain:
#     - "location_id": A unique identifier for this problem location (e.g., "P1", "P2")
#     - "location_description": Brief description of where this problem occurs in the text
#     - "original_text_segment": The corresponding text segment from original text (for comparison)
#     - "fused_text_segment": The specific problematic text from fused text (exact quote)
#     - "problem_category": Problem category (select from: "stylistic_inconsistency", "awkward_expression", "redundant_repetition", "inconsistent_terminology", "tone_mismatch", "sentence_structure_problems", "other_language_issues")
#     - "severity": Severity of THIS specific problem (Minor/Moderate/Severe) - Judge based on impact at this location
#     - "suggested_improvement": Suggested improvement for THIS specific issue
#     - "explanation": Why this is a language quality issue in the fused text compared to original
#     - "comparison_analysis": Brief analysis comparing original and fused versions at this location
#   - "style_consistency_assessment": Overall analysis of style consistency between original and fused texts (Good/Fair/Poor)
#   - "expression_quality_assessment": Overall analysis of expression quality (Natural/Somewhat Awkward/Awkward)
#   - "terminology_consistency_assessment": Overall analysis of terminology consistency between original and fused texts (Consistent/Partially Consistent/Inconsistent)
#   - "sentence_structure_assessment": Overall analysis of sentence structure (Good/Needs Improvement/Poor)
#   - "overall_readability_score": Overall readability score of fused text (1-10, where 10 is best)
#   - "improvements_found": List of language quality improvements in fused text compared to original (if any)
#   - "non_language_issues_noted": List of non-language issues found (factual, logical, etc.) but NOT counted as language problems
#   - "overall_assessment": Overall conclusion about language quality, mentioning the number and distribution of problems found
#   - "final_verdict": Final verdict on whether significant language quality problems exist in fused text compared to original (Yes/No/Partial)

# ## Important Notes:
# - ALWAYS COMPARE FUSED TEXT WITH ORIGINAL TEXT to identify language quality issues.
# - IDENTIFY ALL DISTINCT PROBLEMATIC LOCATIONS in fused text, not just one.
# - FOR EACH LOCATION, report ONLY THE SINGLE MOST SEVERE PROBLEM at that location.
# - APPLY SEVERITY FILTER: Minor issues that do not significantly affect readability may be excluded.
# - ALWAYS provide a complete, structured analysis in the evidence field.
# - The goal is to flag all impactful language issues in fused text compared to original.
# - Use overall_readability_score and problem severity to guide the 'problem_exists' decision.

# Please begin the strict evaluation:
# """

# # ==================== 语言质量评估系统 ====================
# class LanguageQualityEvaluationSystem:
#     class SimpleResponse:
#         def __init__(self, content: str):
#             self.content = content

#     def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, gpu_device: str = "cuda:0"):
#         self.strict_model_config = strict_model_config
#         self.cot_model_config = cot_model_config if cot_model_config else strict_model_config.copy()
#         self.strict_model_name = strict_model_config.get("model", "unknown")
#         self.cot_model_name = self.cot_model_config.get("model", "unknown")
#         self.retry_limit = API_RETRY_LIMIT
#         self.gpu_device = gpu_device
#         self.model_stats = ModelUsageStats()
#         self.model_stats.strict_model_name = self.strict_model_name
#         self.model_stats.cot_model_name = self.cot_model_name
#         if DEBUG_MODE:
#             logger.info(f"🤖 初始化语言质量评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
#             logger.info(f"🎯 指定GPU设备: {gpu_device}")
#             logger.info(f"🎲 使用的随机种子: {RANDOM_SEED}")
        
#         # 设置线程局部随机种子
#         self._set_thread_local_seeds()
        
#         self.llm_strict = self._initialize_llm(strict_model_config)
#         self.llm_cot = self._initialize_llm(self.cot_model_config)

#     def _set_thread_local_seeds(self):
#         """为当前线程设置随机种子"""
#         random.seed(RANDOM_SEED)
#         np.random.seed(RANDOM_SEED)

#     def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
#         temperature = model_config.get("temperature", 0.3)
#         max_tokens = model_config.get("max_tokens", 4000)
#         model_name = model_config.get("model_name", model_config.get("model", "gpt-4"))
#         api_key = model_config.get("api_key", "")
#         base_url = model_config.get("base_url", "https://api.openai.com/v1")
        
#         # 确保温度设置，对于确定性评估可以设置较低温度
#         # 但保持原有温度设置不变
#         if DEBUG_MODE:
#             logger.info(f"🔧 初始化LLM: 模型={model_name}, 温度={temperature}, 最大令牌={max_tokens}, 基础URL={base_url}")
#             logger.info(f"🎲 确保确定性设置: 随机种子={RANDOM_SEED}")
        
#         # 对于vllm，可以尝试设置随机种子参数
#         extra_params = {}
#         if api_key == "EMPTY" or api_key == "vllm":
#             api_key = "vllm"
#             if not base_url or base_url == "https://api.openai.com/v1":
#                 base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
#             logger.info(f"🔧 使用vllm本地模型: {model_name}, 地址: {base_url}")
#             # 为vllm添加随机种子参数（如果支持）
#             extra_params = {"seed": RANDOM_SEED}
#             if DEBUG_MODE:
#                 logger.info(f"🔧 为vLLM设置随机种子: {RANDOM_SEED}")
        
#         # 创建ChatOpenAI实例
#         llm = ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             openai_api_key=api_key,
#             openai_api_base=base_url,
#             request_timeout=REQUEST_TIMEOUT,
#             max_retries=self.retry_limit,
#             **extra_params
#         )
        
#         return llm

#     def estimate_tokens(self, text: str) -> int:
#         chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
#         english_chars = len(re.findall(r'[a-zA-Z]', text))
#         other_chars = len(text) - chinese_chars - english_chars
#         estimated_tokens = int(chinese_chars * 1.5 + english_chars * 0.75 + other_chars * 0.5)
#         return max(estimated_tokens, 1)

#     # 提取文本的通用方法
#     def _extract_text_from_llm_response(self, response_raw: Any) -> str:
#         if response_raw is None:
#             return ""
#         if isinstance(response_raw, str):
#             return response_raw
#         if hasattr(response_raw, "content") and isinstance(getattr(response_raw, "content"), str):
#             return getattr(response_raw, "content")
#         if hasattr(response_raw, "text") and isinstance(getattr(response_raw, "text"), str):
#             return getattr(response_raw, "text")
#         # langchain-like generations
#         try:
#             gens = getattr(response_raw, "generations", None)
#             if gens:
#                 first = gens[0]
#                 if isinstance(first, list) and len(first) > 0:
#                     candidate = first[0]
#                     txt = getattr(candidate, "text", None)
#                     if isinstance(txt, str):
#                         return txt
#                 else:
#                     candidate = first
#                     txt = getattr(candidate, "text", None)
#                     if isinstance(txt, str):
#                         return txt
#         except Exception:
#             pass
#         # openai-like dict
#         if isinstance(response_raw, dict):
#             choices = response_raw.get("choices")
#             if isinstance(choices, list) and len(choices) > 0:
#                 first = choices[0]
#                 if isinstance(first, dict):
#                     msg = first.get("message")
#                     if isinstance(msg, dict) and isinstance(msg.get("content"), str):
#                         return msg.get("content")
#                     if isinstance(first.get("text"), str):
#                         return first.get("text")
#                 for k in ("content", "text", "message"):
#                     if k in response_raw and isinstance(response_raw[k], str):
#                         return response_raw[k]
#         # choices attribute fallback
#         if hasattr(response_raw, "choices"):
#             try:
#                 choices = getattr(response_raw, "choices")
#                 if isinstance(choices, (list, tuple)) and len(choices) > 0:
#                     first = choices[0]
#                     if isinstance(first, dict):
#                         if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
#                             return first["message"]["content"]
#                         if "text" in first and isinstance(first["text"], str):
#                             return first.get("text")
#                     else:
#                         if hasattr(first, "message") and isinstance(getattr(first, "message"), dict):
#                             mc = getattr(first, "message").get("content")
#                             if isinstance(mc, str):
#                                 return mc
#                         if hasattr(first, "text"):
#                             t = getattr(first, "text")
#                             if isinstance(t, str):
#                                 return t
#             except Exception:
#                 pass
#         # fallback to str()
#         try:
#             return str(response_raw)
#         except Exception:
#             return ""

#     def safe_llm_call(self, messages, llm_instance=None, max_retries=3, base_delay=1, step_name: str = "unknown"):
#         llm = llm_instance if llm_instance else self.llm_strict
#         try:
#             if llm_instance is None or llm_instance == self.llm_strict:
#                 self.model_stats.increment_strict()
#             else:
#                 self.model_stats.increment_cot()
#         except Exception:
#             pass
        
#         # 设置重试延迟的确定性种子
#         retry_delay_seed = hash(str(messages) + step_name) % 1000
#         random.seed(RANDOM_SEED + retry_delay_seed)
        
#         for attempt in range(max_retries + 1):
#             try:
#                 response_raw = llm.invoke(messages)
#                 text = self._extract_text_from_llm_response(response_raw)
#                 if DEBUG_MODE:
#                     logger.info(f"📞 模型调用成功: {step_name}, 文本长度: {len(text)}")
#                 return LanguageQualityEvaluationSystem.SimpleResponse(text)
#             except Exception as e:
#                 if attempt < max_retries:
#                     # 使用确定性延迟
#                     delay = base_delay * (2 ** attempt) + (retry_delay_seed % 100) / 1000.0
#                     if DEBUG_MODE:
#                         logger.warning(f"⚠️ API调用失败，第{attempt + 1}次重试: {e}")
#                     time.sleep(delay)
#                 else:
#                     logger.error(f"❌ LLM API 调用最终失败: {e}")
#                     return None

#     # 提取 JSON 块（支持嵌套 {} 或 []）
#     def _extract_json_block(self, text: str) -> Optional[str]:
#         if not text or not isinstance(text, str):
#             return None
#         start_idx = None
#         open_char = None
#         for i, ch in enumerate(text):
#             if ch == '{' or ch == '[':
#                 start_idx = i
#                 open_char = ch
#                 break
#         if start_idx is None:
#             return None
#         close_char = '}' if open_char == '{' else ']'
#         stack = []
#         in_string = False
#         escape = False
#         for j in range(start_idx, len(text)):
#             ch = text[j]
#             if escape:
#                 escape = False
#                 continue
#             if ch == '\\':
#                 escape = True
#                 continue
#             if ch == '"' and not escape:
#                 in_string = not in_string
#                 continue
#             if in_string:
#                 continue
#             if ch == open_char:
#                 stack.append(ch)
#             elif ch == close_char:
#                 if not stack:
#                     return None
#                 stack.pop()
#                 if not stack:
#                     return text[start_idx:j+1]
#         return None

#     def _normalize_parsed(self, parsed: Any) -> Dict[str, Any]:
#         if isinstance(parsed, dict):
#             return parsed
#         if isinstance(parsed, list):
#             for item in parsed:
#                 if isinstance(item, dict):
#                     return item
#             return {"items": parsed}
#         return {"value": parsed}

#     def safe_json_parse(self, response_text: Any) -> Dict[str, Any]:
#         if isinstance(response_text, dict):
#             return response_text
#         if hasattr(response_text, "content"):
#             text = getattr(response_text, "content")
#         else:
#             text = response_text
#         if isinstance(text, str):
#             s = text.strip()
#             try:
#                 parsed = json.loads(s)
#                 return self._normalize_parsed(parsed)
#             except Exception:
#                 json_block = self._extract_json_block(s)
#                 if json_block:
#                     try:
#                         parsed = json.loads(json_block)
#                         return self._normalize_parsed(parsed)
#                     except Exception:
#                         pass
#                 json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s, re.DOTALL)
#                 if json_match:
#                     try:
#                         parsed = json.loads(json_match.group(1))
#                         return self._normalize_parsed(parsed)
#                     except Exception:
#                         pass
#         return {"problem_exists": False, "evidence": {}}

#     # ==================== 语言质量评估 (已重新修改) ====================
#     def evaluate_language_quality(self, problem_type: str, original_text: str, fused_text: str) -> Dict[str, Any]:
#         try:
#             # 设置当前线程的随机种子
#             thread_seed = hash(f"{problem_type}_{original_text[:50]}_{fused_text[:50]}") % 10000
#             random.seed(RANDOM_SEED + thread_seed)
#             np.random.seed(RANDOM_SEED + thread_seed)
            
#             problem_info = LANGUAGE_QUALITY_PROBLEMS.get(problem_type, {})
#             description = problem_info.get("description", "")
#             prompt = LANGUAGE_QUALITY_EVALUATE_PROMPT.format(
#                 problem_type=problem_type,
#                 problem_description=description,
#                 original_text=original_text,
#                 fused_text=fused_text,
#             )
            # messages = [
            #     SystemMessage(content="As a professional news editor and language expert, you are responsible for rigorously evaluating the language quality and style consistency of text fusion. CRITICAL: You MUST compare the fused text with the original text to identify language quality issues. Identify ALL problematic locations in the fused text, and for each location, report ONLY the single most severe language problem compared to original. Apply severity filters. Provide complete structured analysis regardless of whether any problems exist."),
            #     HumanMessage(content=prompt),
            # ]
#             response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"language_quality_{problem_type}")
#             if not response:
#                 return {
#                     "problem_exists": False,
#                     "problem_type": problem_type,
#                     "evidence": {
#                         "analysis_summary": "评估失败，无法获取模型响应",
#                         "identified_problems": [],
#                         "style_consistency_assessment": "Unknown",
#                         "expression_quality_assessment": "Unknown",
#                         "terminology_consistency_assessment": "Unknown",
#                         "sentence_structure_assessment": "Unknown",
#                         "overall_readability_score": 0,
#                         "improvements_found": [],
#                         "non_language_issues_noted": [],
#                         "overall_assessment": "评估失败",
#                         "final_verdict": "No"
#                     },
#                     "evaluation_method": "language_quality_assessment"
#                 }
#             eval_data = self.safe_json_parse(response.content)
#             problem_exists = eval_data.get("problem_exists", False)
#             evidence = eval_data.get("evidence", {})
#             if not evidence:
#                 evidence = {
#                     "analysis_summary": "模型未返回结构化证据",
#                     "identified_problems": [],
#                     "style_consistency_assessment": "Unknown",
#                     "expression_quality_assessment": "Unknown",
#                     "terminology_consistency_assessment": "Unknown",
#                     "sentence_structure_assessment": "Unknown",
#                     "overall_readability_score": 0,
#                     "improvements_found": [],
#                     "non_language_issues_noted": [],
#                     "overall_assessment": "模型未提供评估",
#                     "final_verdict": "No"
#                 }
#             # 确保identified_problems存在并且是列表
#             if "identified_problems" not in evidence or not isinstance(evidence["identified_problems"], list):
#                 evidence["identified_problems"] = []
            
#             # 确保其他必要字段存在
#             for key in ["analysis_summary", "style_consistency_assessment", "expression_quality_assessment", 
#                        "terminology_consistency_assessment", "sentence_structure_assessment", 
#                        "overall_readability_score", "improvements_found", "non_language_issues_noted", 
#                        "overall_assessment", "final_verdict"]:
#                 if key not in evidence:
#                     if key == "identified_problems":
#                         evidence[key] = []
#                     elif key in ["overall_readability_score"]:
#                         evidence[key] = 0
#                     elif key in ["improvements_found", "non_language_issues_noted"]:
#                         evidence[key] = []
#                     else:
#                         evidence[key] = ""
            
#             # 根据identified_problems的数量决定problem_exists
#             if evidence["identified_problems"]:
#                 problem_exists = True
#             else:
#                 problem_exists = False
            
#             return {
#                 "problem_exists": bool(problem_exists),
#                 "problem_type": problem_type,
#                 "evidence": evidence,
#                 "evaluation_method": "language_quality_assessment",
#                 "random_seed_used": RANDOM_SEED + thread_seed
#             }
#         except Exception as e:
#             logger.error(f"❌ 语言质量评估失败 {problem_type}: {e}")
#             return {
#                 "problem_exists": False,
#                 "problem_type": problem_type,
#                 "evidence": {
#                     "analysis_summary": f"评估异常: {str(e)}",
#                     "identified_problems": [],
#                     "style_consistency_assessment": "Unknown",
#                     "expression_quality_assessment": "Unknown",
#                     "terminology_consistency_assessment": "Unknown",
#                     "sentence_structure_assessment": "Unknown",
#                     "overall_readability_score": 0,
#                     "improvements_found": [],
#                     "non_language_issues_noted": [],
#                     "overall_assessment": f"评估异常: {str(e)}",
#                     "final_verdict": "No"
#                 },
#                 "evaluation_method": "language_quality_assessment"
#             }

#     # ==================== 主评估流程 ====================
#     def evaluate(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
#         logger.info("🔄 开始语言质量评估")
#         original_text = news_item.get("input_text", "")
#         fused_text = news_item.get("fused_content", "")
#         if not original_text.strip() or not fused_text.strip():
#             logger.warning(f"❌ 新闻内容为空，无法评估")
#             return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
#         language_quality_results = self.language_quality_evaluation_stage(original_text, fused_text)
        
#         # 计算总问题数（所有identified_problems的总数）
#         total_problems = 0
#         for assessment in language_quality_results["language_assessments"]:
#             if assessment.get("problem_exists", False):
#                 evidence = assessment.get("evidence", {})
#                 identified_problems = evidence.get("identified_problems", [])
#                 total_problems += len(identified_problems)
        
#         language_classification = self._classify_language_problems(language_quality_results["language_errors"])
#         problem_category_summary = self._summarize_language_problem_categories(language_quality_results["language_assessments"])
#         non_language_issues_summary = self._summarize_non_language_issues(language_quality_results["language_assessments"])
        
#         # 新增：详细语言问题分类统计（基于identified_problems）
#         detailed_category_stats = self._generate_detailed_category_statistics(language_quality_results["language_assessments"])
        
#         # 新增：获取改进点
#         improvements_summary = self._summarize_improvements(language_quality_results["language_assessments"])
        
#         result = {
#             "evaluation_success": True,
#             "language_quality_evaluation": language_quality_results,
#             "problem_classification": {
#                 "language_problems": language_classification,
#                 "language_problem_category_summary": problem_category_summary,
#                 "detailed_language_problem_categories": detailed_category_stats,
#                 "non_language_issues_summary": non_language_issues_summary,
#                 "improvements_summary": improvements_summary,
#                 "total_problems": total_problems
#             },
#             "model_usage_stats": self.model_stats.get_stats(),
#             "evaluation_config": {
#                 "random_seed": RANDOM_SEED,
#                 "strict_model": self.strict_model_name,
#                 "cot_model": self.cot_model_name,
#                 "gpu_device": self.gpu_device
#             }
#         }
#         logger.info(f"✅ 语言质量评估完成，共发现 {total_problems} 个语言问题")
#         logger.info(f"🎲 评估配置: 随机种子={RANDOM_SEED}, 严格模型={self.strict_model_name}")
#         return result

#     def language_quality_evaluation_stage(self, original_text: str, fused_text: str) -> Dict[str, Any]:
#         logger.info("🌐 语言质量与风格一致性评估")
#         if not original_text.strip() or not fused_text.strip():
#             return {"language_assessments": [], "language_errors": [], "total_language_problems": 0}
#         if not LANGUAGE_QUALITY_PROBLEMS:
#             logger.info("没有启用的语言质量问题类型，跳过语言质量评估")
#             return {"language_assessments": [], "language_errors": [], "total_language_problems": 0}
#         language_assessments = []
#         language_errors = []
        
#         # 确定性问题类型顺序
#         problem_types = sorted(list(LANGUAGE_QUALITY_PROBLEMS.keys()))
        
#         with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(LANGUAGE_QUALITY_PROBLEMS))) as executor:
#             future_to_problem = {}
#             for i, problem_type in enumerate(problem_types):
#                 # 为每个任务设置确定性种子
#                 task_seed = RANDOM_SEED + i
#                 random.seed(task_seed)
#                 np.random.seed(task_seed)
                
#                 future = executor.submit(self.evaluate_language_quality, problem_type, original_text, fused_text)
#                 future_to_problem[future] = problem_type
            
#             # 按提交顺序获取结果以确保确定性
#             futures = list(future_to_problem.keys())
#             for future in as_completed(futures):
#                 problem_type = future_to_problem[future]
#                 try:
#                     result = future.result()
#                     language_assessments.append({
#                         "problem_type": result["problem_type"],
#                         "problem_exists": result.get("problem_exists", False),
#                         "evidence": result.get("evidence", {}),
#                         "evaluation_method": result.get("evaluation_method", ""),
#                     })
#                     if result.get("problem_exists", False):
#                         language_errors.append({
#                             "error_type": result["problem_type"],
#                             "evidence": result.get("evidence", {}),
#                             "original_text": original_text,
#                             "fused_text": fused_text,
#                         })
#                 except Exception as e:
#                     logger.error(f"❌ 语言质量问题类型评估异常: {problem_type}, {e}")
#                     language_assessments.append({
#                         "problem_type": problem_type,
#                         "problem_exists": False,
#                         "evidence": {
#                             "analysis_summary": f"评估异常: {str(e)}",
#                             "identified_problems": [],
#                             "style_consistency_assessment": "Unknown",
#                             "expression_quality_assessment": "Unknown",
#                             "terminology_consistency_assessment": "Unknown",
#                             "sentence_structure_assessment": "Unknown",
#                             "overall_readability_score": 0,
#                             "improvements_found": [],
#                             "non_language_issues_noted": [],
#                             "overall_assessment": f"评估异常: {str(e)}",
#                             "final_verdict": "No"
#                         },
#                         "evaluation_method": "language_quality_assessment",
#                         "error": str(e)
#                     })
        
#         # 计算总问题数（所有identified_problems的总数）
#         total_problems = 0
#         for assessment in language_assessments:
#             if assessment.get("problem_exists", False):
#                 evidence = assessment.get("evidence", {})
#                 identified_problems = evidence.get("identified_problems", [])
#                 total_problems += len(identified_problems)
        
#         return {
#             "language_assessments": language_assessments,
#             "language_errors": language_errors,
#             "total_language_problems": total_problems
#         }

#     def _classify_language_problems(self, language_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
#         classification = defaultdict(int)
#         for error in language_errors:
#             error_type = error.get("error_type", "未知类型")
#             classification[error_type] += 1
#         return {"by_type": dict(classification), "total_count": len(language_errors)}

#     def _summarize_language_problem_categories(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """汇总语言问题分类统计 (基于identified_problems)"""
#         category_counts = defaultdict(int)
#         all_problems = []
#         for assessment in language_assessments:
#             if assessment.get("problem_exists", False):
#                 evidence = assessment.get("evidence", {})
#                 identified_problems = evidence.get("identified_problems", [])
#                 for problem in identified_problems:
#                     problem_category = problem.get("problem_category", "")
#                     if problem_category:
#                         category_counts[problem_category] += 1
#                         all_problems.append({
#                             "category": problem_category,
#                             "original_text_segment": problem.get("original_text_segment", ""),
#                             "fused_text_segment": problem.get("fused_text_segment", ""),
#                             "severity": problem.get("severity", "Unknown"),
#                             "suggestion": problem.get("suggested_improvement", ""),
#                             "explanation": problem.get("explanation", ""),
#                             "location": problem.get("location_description", ""),
#                             "comparison_analysis": problem.get("comparison_analysis", "")
#                         })
#         category_display = {}
#         for eng_category, chn_name in LANGUAGE_PROBLEM_CATEGORIES.items():
#             if eng_category in category_counts:
#                 category_display[chn_name] = category_counts[eng_category]
#         return {
#             "category_counts": dict(category_counts),
#             "category_display": category_display,
#             "total_categories": len(category_counts),
#             "all_problems": all_problems,
#             "total_problems": sum(category_counts.values())
#         }

#     def _generate_detailed_category_statistics(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """生成详细的7个语言问题分类统计（基于identified_problems）"""
#         detailed_stats = {
#             "categories": {},
#             "total_by_category": defaultdict(int),
#             "severity_distribution": defaultdict(lambda: defaultdict(int)),
#             "examples_by_category": defaultdict(list)
#         }
        
#         # 初始化所有7个类别
#         for category_key, category_name in LANGUAGE_PROBLEM_CATEGORIES.items():
#             detailed_stats["categories"][category_key] = {
#                 "english_name": category_key,
#                 "chinese_name": category_name,
#                 "count": 0,
#                 "severity_counts": {"Minor": 0, "Moderate": 0, "Severe": 0},
#                 "examples": []
#             }
        
#         # 遍历所有identified_problems
#         for assessment in language_assessments:
#             if assessment.get("problem_exists", False):
#                 evidence = assessment.get("evidence", {})
#                 identified_problems = evidence.get("identified_problems", [])
#                 for problem in identified_problems:
#                     problem_category = problem.get("problem_category", "")
#                     if problem_category and problem_category in LANGUAGE_PROBLEM_CATEGORIES:
#                         # 更新类别计数
#                         detailed_stats["categories"][problem_category]["count"] += 1
#                         detailed_stats["total_by_category"][problem_category] += 1
                        
#                         # 获取严重程度
#                         severity = problem.get("severity", "Unknown")
#                         if severity in ["Minor", "Moderate", "Severe"]:
#                             detailed_stats["categories"][problem_category]["severity_counts"][severity] += 1
#                             detailed_stats["severity_distribution"][problem_category][severity] += 1
                        
#                         # 添加示例
#                         fused_text_segment = problem.get("fused_text_segment", "")
#                         original_text_segment = problem.get("original_text_segment", "")
#                         if fused_text_segment:
#                             example_entry = {
#                                 "original_text": original_text_segment[:200] + "..." if len(original_text_segment) > 200 else original_text_segment,
#                                 "fused_text": fused_text_segment[:200] + "..." if len(fused_text_segment) > 200 else fused_text_segment,
#                                 "severity": severity,
#                                 "suggested_improvement": problem.get("suggested_improvement", ""),
#                                 "location_description": problem.get("location_description", ""),
#                                 "explanation": problem.get("explanation", ""),
#                                 "comparison_analysis": problem.get("comparison_analysis", "")
#                             }
#                             detailed_stats["categories"][problem_category]["examples"].append(example_entry)
#                             detailed_stats["examples_by_category"][problem_category].append(example_entry)
        
#         # 转换为可序列化的字典
#         result = {
#             "categories": {},
#             "summary": {
#                 "total_problems": sum(stats["count"] for stats in detailed_stats["categories"].values()),
#                 "category_distribution": {LANGUAGE_PROBLEM_CATEGORIES.get(k, k): v for k, v in detailed_stats["total_by_category"].items()},
#                 "severity_summary": {
#                     "total_by_severity": defaultdict(int)
#                 }
#             }
#         }
        
#         # 填充类别详细信息
#         for category_key, category_info in detailed_stats["categories"].items():
#             result["categories"][category_key] = category_info
            
#             # 更新严重程度汇总
#             for severity, count in category_info["severity_counts"].items():
#                 result["summary"]["severity_summary"]["total_by_severity"][severity] += count
        
#         # 添加中文类别名称映射
#         result["category_mapping"] = LANGUAGE_PROBLEM_CATEGORIES
        
#         return result

#     def _summarize_non_language_issues(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """汇总非语言问题统计"""
#         all_non_language_issues = []
#         for assessment in language_assessments:
#             evidence = assessment.get("evidence", {})
#             non_language_issues = evidence.get("non_language_issues_noted", [])
#             if non_language_issues:
#                 all_non_language_issues.extend(non_language_issues)
#         return {
#             "non_language_issues_count": len(all_non_language_issues),
#             "non_language_issues_list": all_non_language_issues,
#             "has_non_language_issues": len(all_non_language_issues) > 0
#         }

#     def _summarize_improvements(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """汇总融合文本中的改进点"""
#         all_improvements = []
#         for assessment in language_assessments:
#             evidence = assessment.get("evidence", {})
#             improvements = evidence.get("improvements_found", [])
#             if improvements:
#                 all_improvements.extend(improvements)
#         return {
#             "improvements_count": len(all_improvements),
#             "improvements_list": all_improvements,
#             "has_improvements": len(all_improvements) > 0
#         }

#     def get_model_usage_stats(self):
#         return self.model_stats.get_stats()

#     def print_model_usage_summary(self):
#         self.model_stats.print_summary()


# # ==================== 处理器与工具函数 ====================
# def create_language_quality_evaluator(strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
#     if strict_model_name is None:
#         strict_model_name = DEFAULT_TEACHER_MODEL
#     if cot_model_name is None:
#         cot_model_name = strict_model_name
#     strict_model_config = get_model_deployment_config(strict_model_name)
#     cot_model_config = get_model_deployment_config(cot_model_name)
#     return LanguageQualityEvaluationSystem(strict_model_config, cot_model_config, gpu_device=gpu_device)


# class LanguageQualityEvaluationProcessor:
#     def __init__(self, strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
#         self.evaluator = create_language_quality_evaluator(strict_model_name, cot_model_name, gpu_device=gpu_device)
#         self.max_workers = max(1, MAX_WORKERS // 2)
#         self.batch_size = max(1, BATCH_SIZE // 2)
#         self.strict_model_name = strict_model_name or DEFAULT_TEACHER_MODEL
#         self.cot_model_name = cot_model_name or self.strict_model_name
#         logger.info(f"🎲 处理器初始化完成，随机种子: {RANDOM_SEED}")

#     def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             evaluation_result = self.evaluator.evaluate(news_item)
#             return self._create_evaluated_news(news_item, evaluation_result)
#         except Exception as e:
#             logger.error(f"❌ 语言质量评估失败: {e}")
#             return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

#     def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
#         evaluated_news = original_news.copy()
#         evaluated_news.update({
#             "language_quality_evaluation": evaluation_result,
#             "evaluation_success": evaluation_result.get("evaluation_success", False),
#         })
#         if evaluation_result.get("evaluation_success", False):
#             evaluated_news.update({
#                 "language_problem_count": evaluation_result["language_quality_evaluation"]["total_language_problems"],
#                 "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
#                 "improvements_count": evaluation_result["problem_classification"]["improvements_summary"]["improvements_count"],
#             })
#         return evaluated_news

#     def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         logger.info(f"开始批量语言质量评估 {len(news_list)} 篇新闻")
#         logger.info(f"🎲 评估配置: 随机种子={RANDOM_SEED}, 严格模型={self.strict_model_name}, CoT模型={self.cot_model_name}")
        
#         processed_news = []
#         success_count = 0
#         total_language_problems = 0
#         total_improvements = 0
#         batch_size = min(self.batch_size, 2)
        
#         # 确定性批处理顺序
#         for batch_start in range(0, len(news_list), batch_size):
#             batch_end = min(batch_start + batch_size, len(news_list))
#             batch = news_list[batch_start:batch_end]
#             logger.info(f"🔄 评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
#             batch_results = []
#             pbar = tqdm(total=len(batch), desc=f"评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
            
#             # 设置线程池的确定性
#             with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
#                 # 确定性任务提交顺序
#                 future_to_news = {}
#                 for i, news in enumerate(batch):
#                     # 为每个任务设置确定性种子
#                     task_seed = RANDOM_SEED + batch_start + i
#                     random.seed(task_seed)
#                     np.random.seed(task_seed)
                    
#                     future = executor.submit(self.process_single_news, news)
#                     future_to_news[future] = news
                
#                 # 按提交顺序获取结果以确保确定性
#                 futures = list(future_to_news.keys())
#                 for future in as_completed(futures):
#                     news_item = future_to_news[future]
#                     try:
#                         result = future.result()
#                         batch_results.append(result)
#                         if result.get("evaluation_success", False):
#                             success_count += 1
#                             total_language_problems += result.get("language_problem_count", 0)
#                             total_improvements += result.get("improvements_count", 0)
#                     except Exception as e:
#                         logger.error(f"❌ 评估批次处理失败: {e}")
#                         error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
#                         batch_results.append(error_news)
#                     finally:
#                         if pbar:
#                             pbar.update(1)
            
#             if pbar:
#                 pbar.close()
            
#             # 按新闻ID或其他确定性顺序排序结果
#             batch_results_sorted = sorted(batch_results, key=lambda x: x.get("id", 0) if "id" in x else 0)
#             processed_news.extend(batch_results_sorted)
            
#             batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
#             logger.info(f"✅ 评估批次完成: 成功 {batch_success}/{len(batch_results)}")
        
#         self.evaluator.print_model_usage_summary()
#         logger.info(f"🎉 批量语言质量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
#         logger.info(f"📊 总语言质量问题: {total_language_problems}")
#         logger.info(f"📈 总改进点: {total_improvements}")
#         logger.info(f"🎲 评估可复现信息: 随机种子={RANDOM_SEED}")
#         return processed_news


# # ==================== 其他工具函数 ====================
# def load_news_for_evaluation(file_path: str) -> List[Dict[str, Any]]:
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         if isinstance(data, list):
#             logger.info(f"加载 {len(data)} 条新闻（旧格式）")
#             # 确定性排序以确保处理顺序一致
#             data_sorted = sorted(data, key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
#             return data_sorted
#         elif isinstance(data, dict):
#             if "data" in data and isinstance(data["data"], list):
#                 logger.info(f"加载到嵌套结构，顶层data包含 {len(data['data'])} 条记录")
#                 if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
#                     first_item = data["data"][0]
#                     if any(key in first_item for key in ["fused_content", "input_text", "alignment"]):
#                         logger.info(f"直接使用顶层data中的 {len(data['data'])} 条新闻项")
#                         # 确定性排序
#                         data_sorted = sorted(data["data"], key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
#                         return data_sorted
#                     elif "data" in first_item and isinstance(first_item["data"], list):
#                         inner_items = []
#                         for outer_item in data["data"]:
#                             if isinstance(outer_item, dict) and "data" in outer_item and isinstance(outer_item["data"], list):
#                                 inner_items.extend(outer_item["data"])
#                         if inner_items:
#                             logger.info(f"从嵌套结构中提取 {len(inner_items)} 条内层新闻项")
#                             # 确定性排序
#                             inner_sorted = sorted(inner_items, key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
#                             return inner_sorted
#                 logger.warning("输入JSON是对象，包装成列表")
#                 return [data]
#             else:
#                 logger.error("不支持的输入JSON根类型")
#                 return []
#     except Exception as e:
#         logger.error("load_news_for_evaluation失败: %s", e)
#         return []


# def add_model_suffix_to_filename(file_path: str, strict_model_name: str, cot_model_name: str) -> str:
#     safe_strict_name = re.sub(r'[^\w\-]', '_', strict_model_name)
#     safe_cot_name = re.sub(r'[^\w\-]', '_', cot_model_name)
#     if safe_strict_name == safe_cot_name:
#         model_suffix = safe_strict_name
#     else:
#         model_suffix = f"{safe_strict_name}_{safe_cot_name}"
#     dir_name = os.path.dirname(file_path)
#     base_name = os.path.basename(file_path)
#     name_parts = os.path.splitext(base_name)
#     file_name_without_ext = name_parts[0]
#     file_ext = name_parts[1] or ".json"
#     new_base_name = f"{file_name_without_ext}_language_quality_{model_suffix}{file_ext}"
#     return os.path.join(dir_name, new_base_name)


# def save_evaluated_news(news_list: List[Dict[str, Any]], file_path: str, stats: Optional[Dict[str, Any]] = None, strict_model_name: str = "unknown", cot_model_name: str = "unknown"):
#     try:
#         os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
#         output_data = {
#             "evaluation_stats": stats if stats else {},
#             "strict_model": strict_model_name,
#             "cot_model": cot_model_name,
#             "evaluation_type": "language_quality_comparison",
#             "total_items": len(news_list),
#             "random_seed": RANDOM_SEED,
#             "evaluation_timestamp": datetime.now().isoformat(),
#             "data": news_list
#         }
#         with open(file_path, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, ensure_ascii=False, indent=2)
#         logger.info(f"💾 成功保存 {len(news_list)} 篇语言质量评估新闻到 {file_path}")
#         logger.info(f"🎲 保存信息: 随机种子={RANDOM_SEED}")
#     except Exception as e:
#         logger.error(f"❌ 保存文件失败: {e}")


# def generate_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
#     if not successful_evaluations:
#         return {
#             "summary": {
#                 "total_news_evaluated": 0,
#                 "average_language_problems": 0.0,
#                 "average_total_problems": 0.0,
#                 "average_improvements": 0.0,
#             },
#             "language_problems_classification": {"by_type": {}, "total_count": 0},
#             "language_problem_category_summary": {},
#             "detailed_language_problem_categories": {},
#             "non_language_issues_summary": {},
#             "improvements_summary": {},
#             "evaluation_config": {
#                 "random_seed": RANDOM_SEED,
#                 "total_news": len(news_list)
#             }
#         }
    
#     total_news = len(successful_evaluations)
#     language_problems_by_type = defaultdict(int)
#     category_totals = defaultdict(int)
#     detailed_category_stats = defaultdict(lambda: {
#         "count": 0,
#         "severity_counts": {"Minor": 0, "Moderate": 0, "Severe": 0},
#         "examples": []
#     })
#     non_language_issues_total = 0
#     improvements_total = 0
    
#     for news in successful_evaluations:
#         evaluation = news.get("language_quality_evaluation", {})
#         problem_classification = evaluation.get("problem_classification", {})
#         language_problems = problem_classification.get("language_problems", {})
#         for problem_type, count in language_problems.get("by_type", {}).items():
#             language_problems_by_type[problem_type] += count
        
#         # 获取详细分类统计（基于identified_problems）
#         detailed_categories = problem_classification.get("detailed_language_problem_categories", {})
#         if detailed_categories and "categories" in detailed_categories:
#             for category_key, category_info in detailed_categories["categories"].items():
#                 if category_info.get("count", 0) > 0:
#                     category_totals[category_key] += category_info["count"]
#                     detailed_category_stats[category_key]["count"] += category_info["count"]
#                     for severity in ["Minor", "Moderate", "Severe"]:
#                         detailed_category_stats[category_key]["severity_counts"][severity] += category_info.get("severity_counts", {}).get(severity, 0)
#                     # 添加示例（限制数量）
#                     if category_info.get("examples"):
#                         detailed_category_stats[category_key]["examples"].extend(category_info["examples"][:2])  # 每个类别最多2个示例
        
#         non_language_summary = problem_classification.get("non_language_issues_summary", {})
#         non_language_issues_total += non_language_summary.get("non_language_issues_count", 0)
        
#         improvements_summary = problem_classification.get("improvements_summary", {})
#         improvements_total += improvements_summary.get("improvements_count", 0)
    
#     avg_language_problems = sum(news.get("language_problem_count", 0) for news in successful_evaluations) / total_news
#     avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
#     avg_improvements = sum(news.get("improvements_count", 0) for news in successful_evaluations) / total_news
    
#     # 构建详细分类显示
#     detailed_category_display = {}
#     severity_summary = {"Minor": 0, "Moderate": 0, "Severe": 0}
    
#     for category_key, stats_info in detailed_category_stats.items():
#         chn_name = LANGUAGE_PROBLEM_CATEGORIES.get(category_key, category_key)
#         detailed_category_display[chn_name] = {
#             "count": stats_info["count"],
#             "severity_distribution": stats_info["severity_counts"],
#             "example_count": len(stats_info["examples"])
#         }
#         for severity, count in stats_info["severity_counts"].items():
#             severity_summary[severity] += count
    
#     return {
#         "summary": {
#             "total_news_evaluated": total_news,
#             "average_language_problems": round(avg_language_problems, 2),
#             "average_total_problems": round(avg_total_problems, 2),
#             "average_improvements": round(avg_improvements, 2),
#             "total_problems_by_severity": severity_summary
#         },
#         "language_problems_classification": {
#             "by_type": dict(language_problems_by_type),
#             "total_count": sum(language_problems_by_type.values())
#         },
#         "language_problem_category_summary": {
#             "category_totals": dict(category_totals),
#             "category_display": {LANGUAGE_PROBLEM_CATEGORIES.get(k, k): v for k, v in category_totals.items()},
#             "total_categories": len(category_totals),
#             "total_problems": sum(category_totals.values())
#         },
#         "detailed_language_problem_categories": {
#             "categories": dict(detailed_category_display),
#             "category_mapping": LANGUAGE_PROBLEM_CATEGORIES,
#             "severity_summary": severity_summary,
#             "total_problems": sum(category_totals.values())
#         },
#         "non_language_issues_summary": {
#             "total_non_language_issues": non_language_issues_total,
#             "average_non_language_issues": non_language_issues_total / total_news if total_news > 0 else 0
#         },
#         "improvements_summary": {
#             "total_improvements": improvements_total,
#             "average_improvements": improvements_total / total_news if total_news > 0 else 0
#         },
#         "evaluation_config": {
#             "random_seed": RANDOM_SEED,
#             "total_news_processed": len(news_list),
#             "successful_evaluations": total_news
#         }
#     }


# # ==================== 主函数 ====================
# def main():
#     # 声明全局变量
#     global RANDOM_SEED
    
#     parser = argparse.ArgumentParser(description="新闻语言质量评估工具（对照原文评估）")
#     parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
#     parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
#     parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
#     parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
#     parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
#     parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
#     parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"随机种子（默认: {RANDOM_SEED}）")
#     args = parser.parse_args()
    
#     # 使用命令行指定的随机种子（如果提供）
#     if args.seed != RANDOM_SEED:
#         RANDOM_SEED = args.seed
#         set_random_seeds(RANDOM_SEED)
#         logger.info(f"🎲 使用命令行指定的随机种子: {RANDOM_SEED}")
#     else:
#         logger.info(f"🎲 使用默认随机种子: {RANDOM_SEED}")
    
#     if not os.path.exists(args.input):
#         print(f"❌ 输入文件不存在: {args.input}")
#         sys.exit(1)
    
#     print(f"🎲 随机种子: {RANDOM_SEED}")
#     print("📥 加载新闻数据...")
#     news_data = load_news_for_evaluation(args.input)
#     if not news_data:
#         print("❌ 没有加载到新闻数据")
#         sys.exit(1)
    
#     if args.sample > 0:
#         news_data = news_data[:args.sample]
#         print(f"🔬 采样处理前 {args.sample} 篇新闻")
    
#     print("🚀 初始化语言质量评估处理器...")
#     print(f"🎯 使用GPU设备: {args.gpu_device}")
#     processor = LanguageQualityEvaluationProcessor(
#         strict_model_name=args.strict_model,
#         cot_model_name=args.cot_model,
#         gpu_device=args.gpu_device
#     )
    
#     print("🔄 开始语言质量评估（对照原文）...")
#     evaluated_news = processor.process_news_batch(news_data)
#     stats = processor.evaluator.get_model_usage_stats()
#     strict_model_name = processor.strict_model_name
#     cot_model_name = processor.cot_model_name
    
#     output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)
#     print("💾 保存结果...")
#     save_evaluated_news(evaluated_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)
    
#     summary = generate_summary_report(evaluated_news)
    
#     print("\n" + "="*80)
#     print("📊 语言质量评估摘要（对照原文）")
#     print("="*80)
#     print(f"  评估新闻总数: {summary['summary']['total_news_evaluated']}")
#     print(f"  平均每篇新闻语言问题数: {summary['summary']['average_language_problems']:.2f}")
#     print(f"  总语言问题数: {summary['language_problem_category_summary']['total_problems']}")
#     print(f"  总改进点: {summary['improvements_summary']['total_improvements']}")
#     print(f"  平均每篇新闻改进点: {summary['improvements_summary']['average_improvements']:.2f}")
#     print(f"  🎲 随机种子: {RANDOM_SEED}")
    
#     # 打印详细语言问题分类统计
#     print("\n🔍 详细语言问题分类统计:")
#     print("-"*80)
    
#     detailed_categories = summary.get('detailed_language_problem_categories', {})
#     if detailed_categories and 'categories' in detailed_categories:
#         categories_data = detailed_categories['categories']
#         total_problems = detailed_categories.get('total_problems', 0)
        
#         if total_problems > 0:
#             print(f"  总语言问题数: {total_problems}")
#             print(f"  问题严重程度分布:")
#             severity_summary = summary['summary'].get('total_problems_by_severity', {})
#             for severity, count in severity_summary.items():
#                 if count > 0:
#                     percentage = (count / total_problems * 100) if total_problems > 0 else 0
#                     print(f"    {severity}: {count} ({percentage:.1f}%)")
            
#             print("\n  各分类问题统计:")
#             print("  " + "-"*70)
#             print(f"  {'分类':<20} {'数量':<10} {'严重程度分布':<40}")
#             print("  " + "-"*70)
            
#             for chn_name, category_info in categories_data.items():
#                 count = category_info.get('count', 0)
#                 if count > 0:
#                     severity_dist = category_info.get('severity_distribution', {})
#                     severity_str = ", ".join([f"{s}:{c}" for s, c in severity_dist.items() if c > 0])
#                     print(f"  {chn_name:<20} {count:<10} {severity_str:<40}")
            
#             print("  " + "-"*70)
            
#             # 打印问题最多的前3个类别
#             top_categories = sorted(
#                 [(chn_name, info.get('count', 0)) for chn_name, info in categories_data.items()], 
#                 key=lambda x: x[1], 
#                 reverse=True
#             )[:3]
            
#             if top_categories:
#                 print(f"\n  🏆 最常见问题类型:")
#                 for i, (chn_name, count) in enumerate(top_categories, 1):
#                     if count > 0:
#                         percentage = (count / total_problems * 100) if total_problems > 0 else 0
#                         print(f"    {i}. {chn_name}: {count} 处 ({percentage:.1f}%)")
#         else:
#             print("  未发现语言质量问题（对照原文）")
#     else:
#         print("  未发现语言质量问题（对照原文）")
    
#     # 打印非语言问题
#     if summary['non_language_issues_summary']['total_non_language_issues'] > 0:
#         print(f"\n📝 非语言问题统计: {summary['non_language_issues_summary']['total_non_language_issues']} 处")
    
#     # 打印改进点
#     if summary['improvements_summary']['total_improvements'] > 0:
#         print(f"\n📈 语言质量改进点: {summary['improvements_summary']['total_improvements']} 处")
#         print(f"   平均每篇新闻改进点: {summary['improvements_summary']['average_improvements']:.2f}")
    
#     print(f"\n🎲 评估可复现信息:")
#     print(f"   - 随机种子: {RANDOM_SEED}")
#     print(f"   - 严格模型: {strict_model_name}")
#     print(f"   - CoT模型: {cot_model_name}")
#     print(f"   - 输出文件: {output_path_with_suffix}")
    
#     print("="*80)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import json
from threading import Lock
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import numpy as np
import jieba
import random
# 导入配置
from config import (
    get_model_deployment_config,
    DEFAULT_TEACHER_MODEL,
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    LOG_LEVEL,
    DEBUG_MODE,
    ENABLE_PROGRESS_BAR,
    API_RETRY_LIMIT,
    COT_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    BATCH_SIZE,
)

# ==================== 设置随机种子以确保可复现性 ====================
RANDOM_SEED = 42

def set_random_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

set_random_seeds(RANDOM_SEED)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

jieba_cache_dir = os.path.join(os.path.expanduser("~"), ".jieba_cache")
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
if DEBUG_MODE:
    logger.info(f"🔧 设置jieba缓存目录: {jieba_cache_dir}")

# ==================== 模型使用统计类 ====================
class ModelUsageStats:
    def __init__(self):
        self.strict_model_name = ""
        self.cot_model_name = ""
        self.strict_model_calls = 0
        self.cot_model_calls = 0
        self.lock = Lock()

    def increment_strict(self):
        with self.lock:
            self.strict_model_calls += 1

    def increment_cot(self):
        with self.lock:
            self.cot_model_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "strict_model": self.strict_model_name,
                "cot_model": self.cot_model_name,
                "strict_model_calls": self.strict_model_calls,
                "cot_model_calls": self.cot_model_calls,
                "total_calls": self.strict_model_calls + self.cot_model_calls,
                "random_seed": RANDOM_SEED
            }

    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 模型使用统计摘要")
        print("="*60)
        print(f"📅 时间: {stats['timestamp']}")
        print(f"🤖 严格模型: {stats['strict_model']}")
        print(f"🤖 CoT模型: {stats['cot_model']}")
        print(f"📞 严格模型调用次数: {stats['strict_model_calls']}")
        print(f"📞 CoT模型调用次数: {stats['cot_model_calls']}")
        print(f"📞 总调用次数: {stats['total_calls']}")
        print(f"🎲 随机种子: {stats['random_seed']}")
        print("="*60)

# ==================== 语言质量问题定义 ====================
LANGUAGE_QUALITY_PROBLEMS = {
    "Language Quality and Style Consistency": {
        "description": "Evaluate the language quality, style consistency, fluency, and overall readability of the fused text. Check for issues such as stylistic inconsistency, awkward expression, redundancy, inconsistent terminology, tone mismatch, sentence structure problems, or other language quality issues.",
        "evaluation_method": "Analyze the language quality of the fused text from multiple dimensions: style consistency, fluency, terminology usage, sentence structure, and overall readability."
    }
}

# 定义语言问题分类（已移除 cohesion_issues）
LANGUAGE_PROBLEM_CATEGORIES = {
    "stylistic_inconsistency": "风格不一致",
    "awkward_expression": "表达生硬或不自然",
    "redundant_repetition": "冗余重复",
    "inconsistent_terminology": "术语不一致",
    "tone_mismatch": "语气/语调不匹配",
    "sentence_structure_problems": "句子结构问题",
    "other_language_issues": "其他语言问题"
}

# ==================== 语言质量评估Prompt模板 (重新修改) ====================
LANGUAGE_QUALITY_EVALUATE_PROMPT = """
As a professional news editor and language expert, please evaluate the language quality and style consistency of the fused text by COMPARING it with the original text.

## Problem Description: {problem_description}

## CRITICAL EVALUATION RULES:
1. **COMPARE WITH ORIGINAL TEXT**: You must ALWAYS compare the fused text with the original text to identify language quality issues. Look for problems that appear in the fused text but not in the original, or problems that are worse in the fused text.
2. **FOCUS ON THE MOST SEVERE PROBLEM AT EACH TEXT LOCATION**: For each problematic text location, identify and report ONLY the single most significant language quality problem at that location. Different locations can have different primary problems.
3. **DO NOT LIMIT TO ONE PROBLEM PER ARTICLE**: Report ALL distinct problematic locations in the text, each with its own most severe problem. A single article may have multiple language issues at different locations.
4. **Severity Hierarchy (applied per location)**: For each problematic location, prioritize problems as follows (highest to lowest): Sentence Structure Problems, Awkward Expression, Stylistic Inconsistency, Tone Mismatch, Inconsistent Terminology, Redundant Repetition, Other Language Issues.
5. **Ignore Minor Issues**: If the most significant issue at a location is minor and does not substantially impact readability or professionalism, you may choose not to report that location.
6. **Exclude Factual/Literal Issues**: Do NOT include factual omission, logical sequence issues, or factual accuracy as language quality problems.

## EVALUATION REQUIREMENTS:
1. **Compare Thoroughly**: Examine both original and fused texts for Style Consistency, Expression Quality, Terminology Use, Sentence Structure, and Readability.
2. **Identify Changes That Degrade Quality**: Focus on changes from original to fused text that introduce or worsen language problems.
3. **Identify All Problematic Locations**: Find ALL text locations in the fused text where language problems exist (compared to original).
4. **Select Primary Issue Per Location**: For EACH problematic location, determine the single most impactful language problem based on the hierarchy above.
5. **Classify Each Issue**: Assign each identified primary issue to its corresponding category.
6. **Judge Severity Per Issue**: Assess if each issue is severe enough to warrant reporting.

## Text Content:
**Original Text**: {original_text}
**Fused Text**: {fused_text}

## Output Requirements:
- Answer in the same language as the input text
- Provide structured evidence in JSON format ALWAYS, regardless of whether any problems exist

## Output Format: Please return the evaluation in JSON format, including the following fields:
- "problem_exists": Boolean, indicating whether ANY SIGNIFICANT language quality problems exist in the fused text (compared to original).
- "evidence": Structured analysis with the following format:
  - "analysis_summary": Brief summary of the overall language quality assessment, focusing on changes from original to fused.
  - "identified_problems": A LIST of objects, each representing a distinct problematic location in the fused text. Each object should contain:
    - "location_id": A unique identifier for this problem location (e.g., "P1", "P2")
    - "location_description": Brief description of where this problem occurs in the text
    - "original_text_segment": The corresponding text segment from original text (for comparison)
    - "fused_text_segment": The specific problematic text from fused text (exact quote)
    - "problem_category": Problem category (select from: "stylistic_inconsistency", "awkward_expression", "redundant_repetition", "inconsistent_terminology", "tone_mismatch", "sentence_structure_problems", "other_language_issues")
    - "severity": Severity of THIS specific problem (Minor/Moderate/Severe) - Judge based on impact at this location
    - "suggested_improvement": Suggested improvement for THIS specific issue
    - "explanation": Why this is a language quality issue in the fused text compared to original
    - "comparison_analysis": Brief analysis comparing original and fused versions at this location
  - "style_consistency_assessment": Overall analysis of style consistency between original and fused texts (Good/Fair/Poor)
  - "expression_quality_assessment": Overall analysis of expression quality (Natural/Somewhat Awkward/Awkward)
  - "terminology_consistency_assessment": Overall analysis of terminology consistency between original and fused texts (Consistent/Partially Consistent/Inconsistent)
  - "sentence_structure_assessment": Overall analysis of sentence structure (Good/Needs Improvement/Poor)
  - "overall_readability_score": Overall readability score of fused text (1-10, where 10 is best)
  - "improvements_found": List of language quality improvements in fused text compared to original (if any)
  - "non_language_issues_noted": List of non-language issues found (factual, logical, etc.) but NOT counted as language problems
  - "overall_assessment": Overall conclusion about language quality, mentioning the number and distribution of problems found
  - "final_verdict": Final verdict on whether significant language quality problems exist in fused text compared to original (Yes/No/Partial)

## Important Notes:
- ALWAYS COMPARE FUSED TEXT WITH ORIGINAL TEXT to identify language quality issues.
- IDENTIFY ALL DISTINCT PROBLEMATIC LOCATIONS in fused text, not just one.
- FOR EACH LOCATION, report ONLY THE SINGLE MOST SEVERE PROBLEM at that location.
- APPLY SEVERITY FILTER: Minor issues that do not significantly affect readability may be excluded.
- ALWAYS provide a complete, structured analysis in the evidence field.
- The goal is to flag all impactful language issues in fused text compared to original.
- Use overall_readability_score and problem severity to guide the 'problem_exists' decision.

Please begin the strict evaluation:
"""

# ==================== 语言质量评估系统 ====================
class LanguageQualityEvaluationSystem:
    class SimpleResponse:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, gpu_device: str = "cuda:0", max_tokens_override: Optional[int] = None):
        self.strict_model_config = strict_model_config.copy()
        self.cot_model_config = cot_model_config.copy() if cot_model_config else strict_model_config.copy()
        
        if max_tokens_override is not None:
            self.strict_model_config["max_tokens"] = max_tokens_override
            self.cot_model_config["max_tokens"] = max_tokens_override
            if DEBUG_MODE:
                logger.info(f"🔧 覆盖 max_tokens 为 {max_tokens_override}")
        
        self.strict_model_name = self.strict_model_config.get("model", "unknown")
        self.cot_model_name = self.cot_model_config.get("model", "unknown")
        self.retry_limit = API_RETRY_LIMIT
        self.gpu_device = gpu_device
        self.model_stats = ModelUsageStats()
        self.model_stats.strict_model_name = self.strict_model_name
        self.model_stats.cot_model_name = self.cot_model_name
        if DEBUG_MODE:
            logger.info(f"🤖 初始化语言质量评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
            logger.info(f"🎯 指定GPU设备: {gpu_device}")
            logger.info(f"🎲 使用的随机种子: {RANDOM_SEED}")
        
        self._set_thread_local_seeds()
        
        self.llm_strict = self._initialize_llm(self.strict_model_config)
        self.llm_cot = self._initialize_llm(self.cot_model_config)

    def _set_thread_local_seeds(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.3)
        max_tokens = model_config.get("max_tokens", 4000)
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
        
        if DEBUG_MODE:
            logger.info(f"🔧 初始化LLM: 模型={model_name}, 温度={temperature}, 最大令牌={max_tokens}, 基础URL={base_url}")
            logger.info(f"🎲 确保确定性设置: 随机种子={RANDOM_SEED}")
        
        extra_params = {}
        if api_key == "EMPTY" or api_key == "vllm":
            api_key = "vllm"
            if not base_url or base_url == "https://api.openai.com/v1":
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            logger.info(f"🔧 使用vllm本地模型: {model_name}, 地址: {base_url}")
            extra_params = {"seed": RANDOM_SEED}
            if DEBUG_MODE:
                logger.info(f"🔧 为vLLM设置随机种子: {RANDOM_SEED}")
        
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT,
            max_retries=self.retry_limit,
            **extra_params
        )
        return llm

    def estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        estimated_tokens = int(chinese_chars * 1.5 + english_chars * 0.75 + other_chars * 0.5)
        return max(estimated_tokens, 1)

    def _extract_text_from_llm_response(self, response_raw: Any) -> str:
        if response_raw is None:
            return ""
        if isinstance(response_raw, str):
            return response_raw
        if hasattr(response_raw, "content") and isinstance(getattr(response_raw, "content"), str):
            return getattr(response_raw, "content")
        if hasattr(response_raw, "text") and isinstance(getattr(response_raw, "text"), str):
            return getattr(response_raw, "text")
        try:
            gens = getattr(response_raw, "generations", None)
            if gens:
                first = gens[0]
                if isinstance(first, list) and len(first) > 0:
                    candidate = first[0]
                    txt = getattr(candidate, "text", None)
                    if isinstance(txt, str):
                        return txt
                else:
                    candidate = first
                    txt = getattr(candidate, "text", None)
                    if isinstance(txt, str):
                        return txt
        except Exception:
            pass
        if isinstance(response_raw, dict):
            choices = response_raw.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg.get("content")
                    if isinstance(first.get("text"), str):
                        return first.get("text")
                for k in ("content", "text", "message"):
                    if k in response_raw and isinstance(response_raw[k], str):
                        return response_raw[k]
        if hasattr(response_raw, "choices"):
            try:
                choices = getattr(response_raw, "choices")
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                            return first["message"]["content"]
                        if "text" in first and isinstance(first["text"], str):
                            return first.get("text")
                    else:
                        if hasattr(first, "message") and isinstance(getattr(first, "message"), dict):
                            mc = getattr(first, "message").get("content")
                            if isinstance(mc, str):
                                return mc
                        if hasattr(first, "text"):
                            t = getattr(first, "text")
                            if isinstance(t, str):
                                return t
            except Exception:
                pass
        try:
            return str(response_raw)
        except Exception:
            return ""

    def safe_llm_call(self, messages, llm_instance=None, max_retries=3, base_delay=1, step_name: str = "unknown"):
        llm = llm_instance if llm_instance else self.llm_strict
        try:
            if llm_instance is None or llm_instance == self.llm_strict:
                self.model_stats.increment_strict()
            else:
                self.model_stats.increment_cot()
        except Exception:
            pass
        
        retry_delay_seed = hash(str(messages) + step_name) % 1000
        random.seed(RANDOM_SEED + retry_delay_seed)
        
        for attempt in range(max_retries + 1):
            try:
                response_raw = llm.invoke(messages)
                text = self._extract_text_from_llm_response(response_raw)
                if DEBUG_MODE:
                    logger.info(f"📞 模型调用成功: {step_name}, 文本长度: {len(text)}")
                return LanguageQualityEvaluationSystem.SimpleResponse(text)
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + (retry_delay_seed % 100) / 1000.0
                    if DEBUG_MODE:
                        logger.warning(f"⚠️ API调用失败，第{attempt + 1}次重试: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ LLM API 调用最终失败: {e}")
                    return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        start_idx = None
        open_char = None
        for i, ch in enumerate(text):
            if ch == '{' or ch == '[':
                start_idx = i
                open_char = ch
                break
        if start_idx is None:
            return None
        close_char = '}' if open_char == '{' else ']'
        stack = []
        in_string = False
        escape = False
        for j in range(start_idx, len(text)):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                stack.append(ch)
            elif ch == close_char:
                if not stack:
                    return None
                stack.pop()
                if not stack:
                    return text[start_idx:j+1]
        return None

    def _normalize_parsed(self, parsed: Any) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
            return {"items": parsed}
        return {"value": parsed}

    def safe_json_parse(self, response_text: Any) -> Dict[str, Any]:
        if isinstance(response_text, dict):
            return response_text
        if hasattr(response_text, "content"):
            text = getattr(response_text, "content")
        else:
            text = response_text
        if isinstance(text, str):
            s = text.strip()
            try:
                parsed = json.loads(s)
                return self._normalize_parsed(parsed)
            except Exception:
                json_block = self._extract_json_block(s)
                if json_block:
                    try:
                        parsed = json.loads(json_block)
                        return self._normalize_parsed(parsed)
                    except Exception:
                        pass
                json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        return self._normalize_parsed(parsed)
                    except Exception:
                        pass
        return {"problem_exists": False, "evidence": {}}

    def evaluate_language_quality(self, problem_type: str, original_text: str, fused_text: str) -> Dict[str, Any]:
        try:
            thread_seed = hash(f"{problem_type}_{original_text[:50]}_{fused_text[:50]}") % 10000
            random.seed(RANDOM_SEED + thread_seed)
            np.random.seed(RANDOM_SEED + thread_seed)
            
            problem_info = LANGUAGE_QUALITY_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            prompt = LANGUAGE_QUALITY_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                original_text=original_text,
                fused_text=fused_text,
            )
            messages = [
                SystemMessage(content="As a professional news editor and language expert, you are responsible for rigorously evaluating the language quality and style consistency of text fusion. CRITICAL: You MUST compare the fused text with the original text to identify language quality issues. Identify ALL problematic locations in the fused text, and for each location, report ONLY the single most severe language problem compared to original. Apply severity filters. Provide complete structured analysis regardless of whether any problems exist."),
                HumanMessage(content=prompt),
            ]
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"language_quality_{problem_type}")
            if not response:
                return {
                    "problem_exists": False,
                    "problem_type": problem_type,
                    "evidence": {
                        "analysis_summary": "评估失败，无法获取模型响应",
                        "identified_problems": [],
                        "style_consistency_assessment": "Unknown",
                        "expression_quality_assessment": "Unknown",
                        "terminology_consistency_assessment": "Unknown",
                        "sentence_structure_assessment": "Unknown",
                        "overall_readability_score": 0,
                        "improvements_found": [],
                        "non_language_issues_noted": [],
                        "overall_assessment": "评估失败",
                        "final_verdict": "No"
                    },
                    "evaluation_method": "language_quality_assessment"
                }
            eval_data = self.safe_json_parse(response.content)
            problem_exists = eval_data.get("problem_exists", False)
            evidence = eval_data.get("evidence", {})
            if not evidence:
                evidence = {
                    "analysis_summary": "模型未返回结构化证据",
                    "identified_problems": [],
                    "style_consistency_assessment": "Unknown",
                    "expression_quality_assessment": "Unknown",
                    "terminology_consistency_assessment": "Unknown",
                    "sentence_structure_assessment": "Unknown",
                    "overall_readability_score": 0,
                    "improvements_found": [],
                    "non_language_issues_noted": [],
                    "overall_assessment": "模型未提供评估",
                    "final_verdict": "No"
                }
            if "identified_problems" not in evidence or not isinstance(evidence["identified_problems"], list):
                evidence["identified_problems"] = []
            
            for key in ["analysis_summary", "style_consistency_assessment", "expression_quality_assessment", 
                       "terminology_consistency_assessment", "sentence_structure_assessment", 
                       "overall_readability_score", "improvements_found", "non_language_issues_noted", 
                       "overall_assessment", "final_verdict"]:
                if key not in evidence:
                    if key == "identified_problems":
                        evidence[key] = []
                    elif key in ["overall_readability_score"]:
                        evidence[key] = 0
                    elif key in ["improvements_found", "non_language_issues_noted"]:
                        evidence[key] = []
                    else:
                        evidence[key] = ""
            
            if evidence["identified_problems"]:
                problem_exists = True
            else:
                problem_exists = False
            
            return {
                "problem_exists": bool(problem_exists),
                "problem_type": problem_type,
                "evidence": evidence,
                "evaluation_method": "language_quality_assessment",
                "random_seed_used": RANDOM_SEED + thread_seed
            }
        except Exception as e:
            logger.error(f"❌ 语言质量评估失败 {problem_type}: {e}")
            return {
                "problem_exists": False,
                "problem_type": problem_type,
                "evidence": {
                    "analysis_summary": f"评估异常: {str(e)}",
                    "identified_problems": [],
                    "style_consistency_assessment": "Unknown",
                    "expression_quality_assessment": "Unknown",
                    "terminology_consistency_assessment": "Unknown",
                    "sentence_structure_assessment": "Unknown",
                    "overall_readability_score": 0,
                    "improvements_found": [],
                    "non_language_issues_noted": [],
                    "overall_assessment": f"评估异常: {str(e)}",
                    "final_verdict": "No"
                },
                "evaluation_method": "language_quality_assessment"
            }

    def evaluate(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔄 开始语言质量评估")
        original_text = news_item.get("input_text", "")
        fused_text = news_item.get("fused_content", "")
        if not original_text.strip() or not fused_text.strip():
            logger.warning(f"❌ 新闻内容为空，无法评估")
            return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
        language_quality_results = self.language_quality_evaluation_stage(original_text, fused_text)
        
        total_problems = 0
        for assessment in language_quality_results["language_assessments"]:
            if assessment.get("problem_exists", False):
                evidence = assessment.get("evidence", {})
                identified_problems = evidence.get("identified_problems", [])
                total_problems += len(identified_problems)
        
        language_classification = self._classify_language_problems(language_quality_results["language_errors"])
        problem_category_summary = self._summarize_language_problem_categories(language_quality_results["language_assessments"])
        non_language_issues_summary = self._summarize_non_language_issues(language_quality_results["language_assessments"])
        detailed_category_stats = self._generate_detailed_category_statistics(language_quality_results["language_assessments"])
        improvements_summary = self._summarize_improvements(language_quality_results["language_assessments"])
        
        result = {
            "evaluation_success": True,
            "language_quality_evaluation": language_quality_results,
            "problem_classification": {
                "language_problems": language_classification,
                "language_problem_category_summary": problem_category_summary,
                "detailed_language_problem_categories": detailed_category_stats,
                "non_language_issues_summary": non_language_issues_summary,
                "improvements_summary": improvements_summary,
                "total_problems": total_problems
            },
            "model_usage_stats": self.model_stats.get_stats(),
            "evaluation_config": {
                "random_seed": RANDOM_SEED,
                "strict_model": self.strict_model_name,
                "cot_model": self.cot_model_name,
                "gpu_device": self.gpu_device
            }
        }
        logger.info(f"✅ 语言质量评估完成，共发现 {total_problems} 个语言问题")
        logger.info(f"🎲 评估配置: 随机种子={RANDOM_SEED}, 严格模型={self.strict_model_name}")
        return result

    def language_quality_evaluation_stage(self, original_text: str, fused_text: str) -> Dict[str, Any]:
        logger.info("🌐 语言质量与风格一致性评估")
        if not original_text.strip() or not fused_text.strip():
            return {"language_assessments": [], "language_errors": [], "total_language_problems": 0}
        if not LANGUAGE_QUALITY_PROBLEMS:
            logger.info("没有启用的语言质量问题类型，跳过语言质量评估")
            return {"language_assessments": [], "language_errors": [], "total_language_problems": 0}
        language_assessments = []
        language_errors = []
        
        problem_types = sorted(list(LANGUAGE_QUALITY_PROBLEMS.keys()))
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(LANGUAGE_QUALITY_PROBLEMS))) as executor:
            future_to_problem = {}
            for i, problem_type in enumerate(problem_types):
                task_seed = RANDOM_SEED + i
                random.seed(task_seed)
                np.random.seed(task_seed)
                
                future = executor.submit(self.evaluate_language_quality, problem_type, original_text, fused_text)
                future_to_problem[future] = problem_type
            
            futures = list(future_to_problem.keys())
            for future in as_completed(futures):
                problem_type = future_to_problem[future]
                try:
                    result = future.result()
                    language_assessments.append({
                        "problem_type": result["problem_type"],
                        "problem_exists": result.get("problem_exists", False),
                        "evidence": result.get("evidence", {}),
                        "evaluation_method": result.get("evaluation_method", ""),
                    })
                    if result.get("problem_exists", False):
                        language_errors.append({
                            "error_type": result["problem_type"],
                            "evidence": result.get("evidence", {}),
                            "original_text": original_text,
                            "fused_text": fused_text,
                        })
                except Exception as e:
                    logger.error(f"❌ 语言质量问题类型评估异常: {problem_type}, {e}")
                    language_assessments.append({
                        "problem_type": problem_type,
                        "problem_exists": False,
                        "evidence": {
                            "analysis_summary": f"评估异常: {str(e)}",
                            "identified_problems": [],
                            "style_consistency_assessment": "Unknown",
                            "expression_quality_assessment": "Unknown",
                            "terminology_consistency_assessment": "Unknown",
                            "sentence_structure_assessment": "Unknown",
                            "overall_readability_score": 0,
                            "improvements_found": [],
                            "non_language_issues_noted": [],
                            "overall_assessment": f"评估异常: {str(e)}",
                            "final_verdict": "No"
                        },
                        "evaluation_method": "language_quality_assessment",
                        "error": str(e)
                    })
        
        total_problems = 0
        for assessment in language_assessments:
            if assessment.get("problem_exists", False):
                evidence = assessment.get("evidence", {})
                identified_problems = evidence.get("identified_problems", [])
                total_problems += len(identified_problems)
        
        return {
            "language_assessments": language_assessments,
            "language_errors": language_errors,
            "total_language_problems": total_problems
        }

    def _classify_language_problems(self, language_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        classification = defaultdict(int)
        for error in language_errors:
            error_type = error.get("error_type", "未知类型")
            classification[error_type] += 1
        return {"by_type": dict(classification), "total_count": len(language_errors)}

    def _summarize_language_problem_categories(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        category_counts = defaultdict(int)
        all_problems = []
        for assessment in language_assessments:
            if assessment.get("problem_exists", False):
                evidence = assessment.get("evidence", {})
                identified_problems = evidence.get("identified_problems", [])
                for problem in identified_problems:
                    problem_category = problem.get("problem_category", "")
                    if problem_category:
                        category_counts[problem_category] += 1
                        all_problems.append({
                            "category": problem_category,
                            "original_text_segment": problem.get("original_text_segment", ""),
                            "fused_text_segment": problem.get("fused_text_segment", ""),
                            "severity": problem.get("severity", "Unknown"),
                            "suggestion": problem.get("suggested_improvement", ""),
                            "explanation": problem.get("explanation", ""),
                            "location": problem.get("location_description", ""),
                            "comparison_analysis": problem.get("comparison_analysis", "")
                        })
        category_display = {}
        for eng_category, chn_name in LANGUAGE_PROBLEM_CATEGORIES.items():
            if eng_category in category_counts:
                category_display[chn_name] = category_counts[eng_category]
        return {
            "category_counts": dict(category_counts),
            "category_display": category_display,
            "total_categories": len(category_counts),
            "all_problems": all_problems,
            "total_problems": sum(category_counts.values())
        }

    def _generate_detailed_category_statistics(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        detailed_stats = {
            "categories": {},
            "total_by_category": defaultdict(int),
            "severity_distribution": defaultdict(lambda: defaultdict(int)),
            "examples_by_category": defaultdict(list)
        }
        
        for category_key, category_name in LANGUAGE_PROBLEM_CATEGORIES.items():
            detailed_stats["categories"][category_key] = {
                "english_name": category_key,
                "chinese_name": category_name,
                "count": 0,
                "severity_counts": {"Minor": 0, "Moderate": 0, "Severe": 0},
                "examples": []
            }
        
        for assessment in language_assessments:
            if assessment.get("problem_exists", False):
                evidence = assessment.get("evidence", {})
                identified_problems = evidence.get("identified_problems", [])
                for problem in identified_problems:
                    problem_category = problem.get("problem_category", "")
                    if problem_category and problem_category in LANGUAGE_PROBLEM_CATEGORIES:
                        detailed_stats["categories"][problem_category]["count"] += 1
                        detailed_stats["total_by_category"][problem_category] += 1
                        
                        severity = problem.get("severity", "Unknown")
                        if severity in ["Minor", "Moderate", "Severe"]:
                            detailed_stats["categories"][problem_category]["severity_counts"][severity] += 1
                            detailed_stats["severity_distribution"][problem_category][severity] += 1
                        
                        fused_text_segment = problem.get("fused_text_segment", "")
                        original_text_segment = problem.get("original_text_segment", "")
                        if fused_text_segment:
                            example_entry = {
                                "original_text": original_text_segment[:200] + "..." if len(original_text_segment) > 200 else original_text_segment,
                                "fused_text": fused_text_segment[:200] + "..." if len(fused_text_segment) > 200 else fused_text_segment,
                                "severity": severity,
                                "suggested_improvement": problem.get("suggested_improvement", ""),
                                "location_description": problem.get("location_description", ""),
                                "explanation": problem.get("explanation", ""),
                                "comparison_analysis": problem.get("comparison_analysis", "")
                            }
                            detailed_stats["categories"][problem_category]["examples"].append(example_entry)
                            detailed_stats["examples_by_category"][problem_category].append(example_entry)
        
        result = {
            "categories": {},
            "summary": {
                "total_problems": sum(stats["count"] for stats in detailed_stats["categories"].values()),
                "category_distribution": {LANGUAGE_PROBLEM_CATEGORIES.get(k, k): v for k, v in detailed_stats["total_by_category"].items()},
                "severity_summary": {
                    "total_by_severity": defaultdict(int)
                }
            }
        }
        
        for category_key, category_info in detailed_stats["categories"].items():
            result["categories"][category_key] = category_info
            for severity, count in category_info["severity_counts"].items():
                result["summary"]["severity_summary"]["total_by_severity"][severity] += count
        
        result["category_mapping"] = LANGUAGE_PROBLEM_CATEGORIES
        return result

    def _summarize_non_language_issues(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_non_language_issues = []
        for assessment in language_assessments:
            evidence = assessment.get("evidence", {})
            non_language_issues = evidence.get("non_language_issues_noted", [])
            if non_language_issues:
                all_non_language_issues.extend(non_language_issues)
        return {
            "non_language_issues_count": len(all_non_language_issues),
            "non_language_issues_list": all_non_language_issues,
            "has_non_language_issues": len(all_non_language_issues) > 0
        }

    def _summarize_improvements(self, language_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_improvements = []
        for assessment in language_assessments:
            evidence = assessment.get("evidence", {})
            improvements = evidence.get("improvements_found", [])
            if improvements:
                all_improvements.extend(improvements)
        return {
            "improvements_count": len(all_improvements),
            "improvements_list": all_improvements,
            "has_improvements": len(all_improvements) > 0
        }

    def get_model_usage_stats(self):
        return self.model_stats.get_stats()

    def print_model_usage_summary(self):
        self.model_stats.print_summary()


# ==================== 处理器与工具函数 ====================
def create_language_quality_evaluator(strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0", max_tokens_override: Optional[int] = None):
    if strict_model_name is None:
        strict_model_name = DEFAULT_TEACHER_MODEL
    if cot_model_name is None:
        cot_model_name = strict_model_name
    strict_model_config = get_model_deployment_config(strict_model_name)
    cot_model_config = get_model_deployment_config(cot_model_name)
    return LanguageQualityEvaluationSystem(strict_model_config, cot_model_config, gpu_device=gpu_device, max_tokens_override=max_tokens_override)


class LanguageQualityEvaluationProcessor:
    def __init__(self, strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0", max_tokens_override: Optional[int] = None):
        self.evaluator = create_language_quality_evaluator(strict_model_name, cot_model_name, gpu_device=gpu_device, max_tokens_override=max_tokens_override)
        self.max_workers = max(1, MAX_WORKERS // 2)
        self.batch_size = max(1, BATCH_SIZE // 2)
        self.strict_model_name = strict_model_name or DEFAULT_TEACHER_MODEL
        self.cot_model_name = cot_model_name or self.strict_model_name
        logger.info(f"🎲 处理器初始化完成，随机种子: {RANDOM_SEED}")

    def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            evaluation_result = self.evaluator.evaluate(news_item)
            return self._create_evaluated_news(news_item, evaluation_result)
        except Exception as e:
            logger.error(f"❌ 语言质量评估失败: {e}")
            return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

    def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluated_news = original_news.copy()
        evaluated_news.update({
            "language_quality_evaluation": evaluation_result,
            "evaluation_success": evaluation_result.get("evaluation_success", False),
        })
        if evaluation_result.get("evaluation_success", False):
            evaluated_news.update({
                "language_problem_count": evaluation_result["language_quality_evaluation"]["total_language_problems"],
                "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
                "improvements_count": evaluation_result["problem_classification"]["improvements_summary"]["improvements_count"],
            })
        return evaluated_news

    def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始批量语言质量评估 {len(news_list)} 篇新闻")
        logger.info(f"🎲 评估配置: 随机种子={RANDOM_SEED}, 严格模型={self.strict_model_name}, CoT模型={self.cot_model_name}")
        
        processed_news = []
        success_count = 0
        total_language_problems = 0
        total_improvements = 0
        batch_size = min(self.batch_size, 2)
        
        for batch_start in range(0, len(news_list), batch_size):
            batch_end = min(batch_start + batch_size, len(news_list))
            batch = news_list[batch_start:batch_end]
            logger.info(f"🔄 评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
            batch_results = []
            pbar = tqdm(total=len(batch), desc=f"评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                future_to_news = {}
                for i, news in enumerate(batch):
                    task_seed = RANDOM_SEED + batch_start + i
                    random.seed(task_seed)
                    np.random.seed(task_seed)
                    
                    future = executor.submit(self.process_single_news, news)
                    future_to_news[future] = news
                
                futures = list(future_to_news.keys())
                for future in as_completed(futures):
                    news_item = future_to_news[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        if result.get("evaluation_success", False):
                            success_count += 1
                            total_language_problems += result.get("language_problem_count", 0)
                            total_improvements += result.get("improvements_count", 0)
                    except Exception as e:
                        logger.error(f"❌ 评估批次处理失败: {e}")
                        error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
                        batch_results.append(error_news)
                    finally:
                        if pbar:
                            pbar.update(1)
            
            if pbar:
                pbar.close()
            
            batch_results_sorted = sorted(batch_results, key=lambda x: x.get("id", 0) if "id" in x else 0)
            processed_news.extend(batch_results_sorted)
            
            batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
            logger.info(f"✅ 评估批次完成: 成功 {batch_success}/{len(batch_results)}")
        
        self.evaluator.print_model_usage_summary()
        logger.info(f"🎉 批量语言质量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
        logger.info(f"📊 总语言质量问题: {total_language_problems}")
        logger.info(f"📈 总改进点: {total_improvements}")
        logger.info(f"🎲 评估可复现信息: 随机种子={RANDOM_SEED}")
        return processed_news


# ==================== 其他工具函数 ====================
def load_news_for_evaluation(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info(f"加载 {len(data)} 条新闻（旧格式）")
            data_sorted = sorted(data, key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
            return data_sorted
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                logger.info(f"加载到嵌套结构，顶层data包含 {len(data['data'])} 条记录")
                if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
                    first_item = data["data"][0]
                    if any(key in first_item for key in ["fused_content", "input_text", "alignment"]):
                        logger.info(f"直接使用顶层data中的 {len(data['data'])} 条新闻项")
                        data_sorted = sorted(data["data"], key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
                        return data_sorted
                    elif "data" in first_item and isinstance(first_item["data"], list):
                        inner_items = []
                        for outer_item in data["data"]:
                            if isinstance(outer_item, dict) and "data" in outer_item and isinstance(outer_item["data"], list):
                                inner_items.extend(outer_item["data"])
                        if inner_items:
                            logger.info(f"从嵌套结构中提取 {len(inner_items)} 条内层新闻项")
                            inner_sorted = sorted(inner_items, key=lambda x: x.get("id", 0) if isinstance(x, dict) and "id" in x else 0)
                            return inner_sorted
                logger.warning("输入JSON是对象，包装成列表")
                return [data]
            else:
                logger.error("不支持的输入JSON根类型")
                return []
    except Exception as e:
        logger.error("load_news_for_evaluation失败: %s", e)
        return []


def add_model_suffix_to_filename(file_path: str, strict_model_name: str, cot_model_name: str) -> str:
    safe_strict_name = re.sub(r'[^\w\-]', '_', strict_model_name)
    safe_cot_name = re.sub(r'[^\w\-]', '_', cot_model_name)
    if safe_strict_name == safe_cot_name:
        model_suffix = safe_strict_name
    else:
        model_suffix = f"{safe_strict_name}_{safe_cot_name}"
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_parts = os.path.splitext(base_name)
    file_name_without_ext = name_parts[0]
    file_ext = name_parts[1] or ".json"
    new_base_name = f"{file_name_without_ext}_language_quality_{model_suffix}{file_ext}"
    return os.path.join(dir_name, new_base_name)


def save_evaluated_news(news_list: List[Dict[str, Any]], file_path: str, stats: Optional[Dict[str, Any]] = None, strict_model_name: str = "unknown", cot_model_name: str = "unknown"):
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        output_data = {
            "evaluation_stats": stats if stats else {},
            "strict_model": strict_model_name,
            "cot_model": cot_model_name,
            "evaluation_type": "language_quality_comparison",
            "total_items": len(news_list),
            "random_seed": RANDOM_SEED,
            "evaluation_timestamp": datetime.now().isoformat(),
            "data": news_list
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 成功保存 {len(news_list)} 篇语言质量评估新闻到 {file_path}")
        logger.info(f"🎲 保存信息: 随机种子={RANDOM_SEED}")
    except Exception as e:
        logger.error(f"❌ 保存文件失败: {e}")


def generate_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
    if not successful_evaluations:
        return {
            "summary": {
                "total_news_evaluated": 0,
                "average_language_problems": 0.0,
                "average_total_problems": 0.0,
                "average_improvements": 0.0,
            },
            "language_problems_classification": {"by_type": {}, "total_count": 0},
            "language_problem_category_summary": {},
            "detailed_language_problem_categories": {},
            "non_language_issues_summary": {},
            "improvements_summary": {},
            "evaluation_config": {
                "random_seed": RANDOM_SEED,
                "total_news": len(news_list)
            }
        }
    
    total_news = len(successful_evaluations)
    language_problems_by_type = defaultdict(int)
    category_totals = defaultdict(int)
    detailed_category_stats = defaultdict(lambda: {
        "count": 0,
        "severity_counts": {"Minor": 0, "Moderate": 0, "Severe": 0},
        "examples": []
    })
    non_language_issues_total = 0
    improvements_total = 0
    
    for news in successful_evaluations:
        evaluation = news.get("language_quality_evaluation", {})
        problem_classification = evaluation.get("problem_classification", {})
        language_problems = problem_classification.get("language_problems", {})
        for problem_type, count in language_problems.get("by_type", {}).items():
            language_problems_by_type[problem_type] += count
        
        detailed_categories = problem_classification.get("detailed_language_problem_categories", {})
        if detailed_categories and "categories" in detailed_categories:
            for category_key, category_info in detailed_categories["categories"].items():
                if category_info.get("count", 0) > 0:
                    category_totals[category_key] += category_info["count"]
                    detailed_category_stats[category_key]["count"] += category_info["count"]
                    for severity in ["Minor", "Moderate", "Severe"]:
                        detailed_category_stats[category_key]["severity_counts"][severity] += category_info.get("severity_counts", {}).get(severity, 0)
                    if category_info.get("examples"):
                        detailed_category_stats[category_key]["examples"].extend(category_info["examples"][:2])
        
        non_language_summary = problem_classification.get("non_language_issues_summary", {})
        non_language_issues_total += non_language_summary.get("non_language_issues_count", 0)
        
        improvements_summary = problem_classification.get("improvements_summary", {})
        improvements_total += improvements_summary.get("improvements_count", 0)
    
    avg_language_problems = sum(news.get("language_problem_count", 0) for news in successful_evaluations) / total_news
    avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
    avg_improvements = sum(news.get("improvements_count", 0) for news in successful_evaluations) / total_news
    
    detailed_category_display = {}
    severity_summary = {"Minor": 0, "Moderate": 0, "Severe": 0}
    
    for category_key, stats_info in detailed_category_stats.items():
        chn_name = LANGUAGE_PROBLEM_CATEGORIES.get(category_key, category_key)
        detailed_category_display[chn_name] = {
            "count": stats_info["count"],
            "severity_distribution": stats_info["severity_counts"],
            "example_count": len(stats_info["examples"])
        }
        for severity, count in stats_info["severity_counts"].items():
            severity_summary[severity] += count
    
    return {
        "summary": {
            "total_news_evaluated": total_news,
            "average_language_problems": round(avg_language_problems, 2),
            "average_total_problems": round(avg_total_problems, 2),
            "average_improvements": round(avg_improvements, 2),
            "total_problems_by_severity": severity_summary
        },
        "language_problems_classification": {
            "by_type": dict(language_problems_by_type),
            "total_count": sum(language_problems_by_type.values())
        },
        "language_problem_category_summary": {
            "category_totals": dict(category_totals),
            "category_display": {LANGUAGE_PROBLEM_CATEGORIES.get(k, k): v for k, v in category_totals.items()},
            "total_categories": len(category_totals),
            "total_problems": sum(category_totals.values())
        },
        "detailed_language_problem_categories": {
            "categories": dict(detailed_category_display),
            "category_mapping": LANGUAGE_PROBLEM_CATEGORIES,
            "severity_summary": severity_summary,
            "total_problems": sum(category_totals.values())
        },
        "non_language_issues_summary": {
            "total_non_language_issues": non_language_issues_total,
            "average_non_language_issues": non_language_issues_total / total_news if total_news > 0 else 0
        },
        "improvements_summary": {
            "total_improvements": improvements_total,
            "average_improvements": improvements_total / total_news if total_news > 0 else 0
        },
        "evaluation_config": {
            "random_seed": RANDOM_SEED,
            "total_news_processed": len(news_list),
            "successful_evaluations": total_news
        }
    }


# ==================== 修复模式辅助函数（核心修改） ====================
def _deep_find_assessments(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    递归查找 language_assessments 列表。
    支持两种结构：
    1. obj 直接包含 "language_assessments" 键
    2. obj 包含 "language_quality_evaluation" 键，且该值包含 "language_assessments"
    """
    if not isinstance(obj, dict):
        return None
    # 直接查找
    if "language_assessments" in obj:
        return obj.get("language_assessments")
    # 嵌套一层
    inner = obj.get("language_quality_evaluation")
    if isinstance(inner, dict):
        if "language_assessments" in inner:
            return inner.get("language_assessments")
        # 继续递归（防止更深嵌套）
        return _deep_find_assessments(inner)
    return None


def is_evaluation_failed(news_item: Dict[str, Any]) -> bool:
    """判断一条新闻的评估是否因未返回结构化证据而失败"""
    # 先尝试找到 language_assessments
    assessments = _deep_find_assessments(news_item)
    if not assessments:
        # 如果没有 assessments，可能整个评估都没有成功
        if not news_item.get("evaluation_success", False):
            return True
        return False
    
    # 检查每个 assessment 的 evidence.analysis_summary
    for assessment in assessments:
        evidence = assessment.get("evidence", {})
        analysis_summary = evidence.get("analysis_summary", "")
        if "模型未返回结构化证据" in analysis_summary:
            return True
    return False


def _get_inner_evaluation_container(news_item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    返回内部评估容器的引用以及路径标识。
    返回 (container, path) 其中 container 是应该被替换的字典，
    path 用于调试。
    如果找不到合适的容器，返回 (None, None)
    """
    # 情况1：news_item 直接包含 "language_quality_evaluation" 且该值有 "language_assessments"
    lqe = news_item.get("language_quality_evaluation")
    if isinstance(lqe, dict):
        if "language_assessments" in lqe:
            return lqe, "direct"
        # 情况2：lqe 内还有一层 "language_quality_evaluation"
        inner_lqe = lqe.get("language_quality_evaluation")
        if isinstance(inner_lqe, dict) and "language_assessments" in inner_lqe:
            return inner_lqe, "nested"
    # 情况3：news_item 本身包含 "language_assessments"
    if "language_assessments" in news_item:
        return news_item, "root"
    return None, None


def fix_evaluation_results(input_file: str, strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
    """修复评估结果文件中未返回结构化证据的条目"""
    print("\n🔧 进入修复模式，将重新评估未返回结构化证据的新闻项")
    print(f"📂 读取评估文件: {input_file}")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        sys.exit(1)
    
    # 提取数据列表
    if isinstance(full_data, list):
        news_list = full_data
        metadata = {}
    elif isinstance(full_data, dict):
        news_list = full_data.get("data", [])
        metadata = {k: v for k, v in full_data.items() if k != "data"}
        if not news_list:
            print("❌ 文件中没有找到 data 字段或数据为空")
            sys.exit(1)
    else:
        print("❌ 不支持的文件格式")
        sys.exit(1)
    
    # 如果没有提供模型名称，尝试从文件中读取
    if strict_model_name is None:
        strict_model_name = metadata.get("strict_model", DEFAULT_TEACHER_MODEL)
    if cot_model_name is None:
        cot_model_name = metadata.get("cot_model", strict_model_name)
    
    # 统计需要修复的条目
    failed_indices = []
    for idx, item in enumerate(news_list):
        if is_evaluation_failed(item):
            failed_indices.append(idx)
    
    before_count = len(failed_indices)
    print(f"📊 修复前: {before_count} 条新闻未返回结构化证据")
    
    if before_count == 0:
        print("✅ 没有需要修复的条目，直接退出")
        return
    
    # 创建评估器（使用 max_tokens=12288）
    print(f"🚀 初始化评估器 (max_tokens=12288), 严格模型: {strict_model_name}, CoT模型: {cot_model_name}")
    evaluator = create_language_quality_evaluator(
        strict_model_name=strict_model_name,
        cot_model_name=cot_model_name,
        gpu_device=gpu_device,
        max_tokens_override=12288
    )
    
    # 逐个重新评估失败的条目
    success_count = 0
    for idx in tqdm(failed_indices, desc="重新评估失败条目"):
        item = news_list[idx]
        original_text = item.get("input_text", "")
        fused_text = item.get("fused_content", "")
        if not original_text.strip() or not fused_text.strip():
            print(f"⚠️ 新闻 {idx} 缺少原始文本或融合文本，跳过")
            continue
        
        try:
            # 执行评估
            eval_result = evaluator.evaluate(item)
            # 获取内部容器并更新
            container, path_type = _get_inner_evaluation_container(item)
            if container is not None:
                # 替换整个评估内容
                # 注意：evaluate 返回的结果已经包含了 evaluation_success, language_quality_evaluation 等字段
                # 但我们需要保持原结构。原结构可能是：
                # - 直接：container 就是 language_quality_evaluation 对象，它应该被替换为 eval_result
                # - 嵌套：container 是内层的 language_quality_evaluation，外层还有一层包装
                # 为了简单，我们直接更新 container 的所有内容，但保留外层结构
                if path_type == "nested":
                    # 外层 lqe 有 evaluation_success 字段，内层需要完全替换
                    # 我们保持外层不变，只替换内层
                    container.clear()
                    container.update(eval_result)
                else:
                    # 直接模式：container 就是整个评估结果，直接替换
                    container.clear()
                    container.update(eval_result)
            else:
                # 找不到容器，则直接替换顶层 language_quality_evaluation
                # 但这种情况很少见，我们仍然尝试
                item["language_quality_evaluation"] = eval_result
            
            # 更新顶层便捷字段（如果存在）
            if eval_result.get("evaluation_success", False):
                item["evaluation_success"] = True
                item["language_problem_count"] = eval_result.get("language_quality_evaluation", {}).get("total_language_problems", 0)
                item["total_problem_count"] = eval_result.get("problem_classification", {}).get("total_problems", 0)
                item["improvements_count"] = eval_result.get("problem_classification", {}).get("improvements_summary", {}).get("improvements_count", 0)
            else:
                item["evaluation_success"] = False
            
            success_count += 1
        except Exception as e:
            logger.error(f"❌ 重新评估新闻 {idx} 失败: {e}")
    
    # 重新统计修复后的失败数量
    after_failed = sum(1 for item in news_list if is_evaluation_failed(item))
    after_count = after_failed
    print(f"\n📊 修复后: {after_count} 条新闻仍未返回结构化证据")
    print(f"✅ 成功修复: {before_count - after_count} 条")
    
    # 保存更新后的文件（覆盖原文件）
    output_data = metadata.copy()
    output_data["data"] = news_list
    output_data["fix_timestamp"] = datetime.now().isoformat()
    output_data["fix_stats"] = {
        "before_count": before_count,
        "after_count": after_count,
        "fixed_count": before_count - after_count,
        "max_tokens_used": 12288
    }
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"💾 已更新原文件: {input_file}")


# ==================== 主函数 ====================
def main():
    global RANDOM_SEED
    
    parser = argparse.ArgumentParser(description="新闻语言质量评估工具（对照原文评估）")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=False, help="输出JSON文件路径（正常模式必需，修复模式下忽略）")
    parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
    parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
    parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
    parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"随机种子（默认: {RANDOM_SEED}）")
    parser.add_argument("--fix-mode", action="store_true", help="修复模式：重新评估未返回结构化证据的条目，更新原文件")
    args = parser.parse_args()
    
    if args.seed != RANDOM_SEED:
        RANDOM_SEED = args.seed
        set_random_seeds(RANDOM_SEED)
        logger.info(f"🎲 使用命令行指定的随机种子: {RANDOM_SEED}")
    else:
        logger.info(f"🎲 使用默认随机种子: {RANDOM_SEED}")
    
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 修复模式
    if args.fix_mode:
        fix_evaluation_results(
            input_file=args.input,
            strict_model_name=args.strict_model,
            cot_model_name=args.cot_model,
            gpu_device=args.gpu_device
        )
        return
    
    # 正常模式
    if not args.output:
        print("❌ 正常模式下必须指定 --output 输出文件路径")
        sys.exit(1)
    
    print(f"🎲 随机种子: {RANDOM_SEED}")
    print("📥 加载新闻数据...")
    news_data = load_news_for_evaluation(args.input)
    if not news_data:
        print("❌ 没有加载到新闻数据")
        sys.exit(1)
    
    if args.sample > 0:
        news_data = news_data[:args.sample]
        print(f"🔬 采样处理前 {args.sample} 篇新闻")
    
    print("🚀 初始化语言质量评估处理器...")
    print(f"🎯 使用GPU设备: {args.gpu_device}")
    processor = LanguageQualityEvaluationProcessor(
        strict_model_name=args.strict_model,
        cot_model_name=args.cot_model,
        gpu_device=args.gpu_device
    )
    
    print("🔄 开始语言质量评估（对照原文）...")
    evaluated_news = processor.process_news_batch(news_data)
    stats = processor.evaluator.get_model_usage_stats()
    strict_model_name = processor.strict_model_name
    cot_model_name = processor.cot_model_name
    
    output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)
    print("💾 保存结果...")
    save_evaluated_news(evaluated_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)
    
    summary = generate_summary_report(evaluated_news)
    
    print("\n" + "="*80)
    print("📊 语言质量评估摘要（对照原文）")
    print("="*80)
    print(f"  评估新闻总数: {summary['summary']['total_news_evaluated']}")
    print(f"  平均每篇新闻语言问题数: {summary['summary']['average_language_problems']:.2f}")
    print(f"  总语言问题数: {summary['language_problem_category_summary']['total_problems']}")
    print(f"  总改进点: {summary['improvements_summary']['total_improvements']}")
    print(f"  平均每篇新闻改进点: {summary['improvements_summary']['average_improvements']:.2f}")
    print(f"  🎲 随机种子: {RANDOM_SEED}")
    
    print("\n🔍 详细语言问题分类统计:")
    print("-"*80)
    
    detailed_categories = summary.get('detailed_language_problem_categories', {})
    if detailed_categories and 'categories' in detailed_categories:
        categories_data = detailed_categories['categories']
        total_problems = detailed_categories.get('total_problems', 0)
        
        if total_problems > 0:
            print(f"  总语言问题数: {total_problems}")
            print(f"  问题严重程度分布:")
            severity_summary = summary['summary'].get('total_problems_by_severity', {})
            for severity, count in severity_summary.items():
                if count > 0:
                    percentage = (count / total_problems * 100) if total_problems > 0 else 0
                    print(f"    {severity}: {count} ({percentage:.1f}%)")
            
            print("\n  各分类问题统计:")
            print("  " + "-"*70)
            print(f"  {'分类':<20} {'数量':<10} {'严重程度分布':<40}")
            print("  " + "-"*70)
            
            for chn_name, category_info in categories_data.items():
                count = category_info.get('count', 0)
                if count > 0:
                    severity_dist = category_info.get('severity_distribution', {})
                    severity_str = ", ".join([f"{s}:{c}" for s, c in severity_dist.items() if c > 0])
                    print(f"  {chn_name:<20} {count:<10} {severity_str:<40}")
            
            print("  " + "-"*70)
            
            top_categories = sorted(
                [(chn_name, info.get('count', 0)) for chn_name, info in categories_data.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            if top_categories:
                print(f"\n  🏆 最常见问题类型:")
                for i, (chn_name, count) in enumerate(top_categories, 1):
                    if count > 0:
                        percentage = (count / total_problems * 100) if total_problems > 0 else 0
                        print(f"    {i}. {chn_name}: {count} 处 ({percentage:.1f}%)")
        else:
            print("  未发现语言质量问题（对照原文）")
    else:
        print("  未发现语言质量问题（对照原文）")
    
    if summary['non_language_issues_summary']['total_non_language_issues'] > 0:
        print(f"\n📝 非语言问题统计: {summary['non_language_issues_summary']['total_non_language_issues']} 处")
    
    if summary['improvements_summary']['total_improvements'] > 0:
        print(f"\n📈 语言质量改进点: {summary['improvements_summary']['total_improvements']} 处")
        print(f"   平均每篇新闻改进点: {summary['improvements_summary']['average_improvements']:.2f}")
    
    print(f"\n🎲 评估可复现信息:")
    print(f"   - 随机种子: {RANDOM_SEED}")
    print(f"   - 严格模型: {strict_model_name}")
    print(f"   - CoT模型: {cot_model_name}")
    print(f"   - 输出文件: {output_path_with_suffix}")
    
    print("="*80)


if __name__ == "__main__":
    main()