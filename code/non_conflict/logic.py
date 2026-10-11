# # #!/usr/bin/env python3
# # import subprocess
# # import sys
# # import os
# # import re
# # import json
# # from threading import Lock
# # import time
# # from datetime import datetime
# # from typing import List, Dict, Any, Optional, Tuple
# # from langchain_community.chat_models import ChatOpenAI
# # from langchain.schema import SystemMessage, HumanMessage
# # import logging
# # from collections import Counter, defaultdict
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from tqdm import tqdm
# # import argparse
# # import numpy as np
# # import jieba
# # import random
# # import torch

# # # 导入配置（请确保你的 config.py 可用并提供所需变量/函数）
# # from config import (
# #     get_model_deployment_config,
# #     DEFAULT_TEACHER_MODEL,
# #     MAX_WORKERS,
# #     REQUEST_TIMEOUT,
# #     LOG_LEVEL,
# #     DEBUG_MODE,
# #     ENABLE_PROGRESS_BAR,
# #     API_RETRY_LIMIT,
# #     COT_CONFIDENCE_THRESHOLD,
# #     LOW_CONFIDENCE_THRESHOLD,
# #     BATCH_SIZE,
# # )

# # # ==================== 设置全局随机种子以确保可复现性 ====================
# # def set_global_seed(seed: int = 42):
# #     """设置所有可能的随机种子以确保评估的可复现性"""
# #     print(f"🔧 设置全局随机种子: {seed}")
    
# #     # 设置Python内置随机种子
# #     random.seed(seed)
    
# #     # 设置numpy随机种子
# #     np.random.seed(seed)
    
# #     # 设置torch随机种子（如果可用）
# #     try:
# #         torch.manual_seed(seed)
# #         if torch.cuda.is_available():
# #             torch.cuda.manual_seed(seed)
# #             torch.cuda.manual_seed_all(seed)
# #             torch.backends.cudnn.deterministic = True
# #             torch.backends.cudnn.benchmark = False
# #     except ImportError:
# #         pass
    
# #     # 设置环境变量
# #     os.environ['PYTHONHASHSEED'] = str(seed)
    
# #     print(f"✅ 全局随机种子已设置为 {seed}")

# # # 设置默认随机种子（可在命令行参数中覆盖）
# # DEFAULT_SEED = 42
# # set_global_seed(DEFAULT_SEED)

# # # 日志配置
# # logging.basicConfig(level=getattr(logging, LOG_LEVEL))
# # logger = logging.getLogger(__name__)

# # # 修复jieba缓存目录问题
# # jieba_cache_dir = os.path.join(os.path.expanduser("~"), ".jieba_cache")
# # os.makedirs(jieba_cache_dir, exist_ok=True)
# # jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
# # if DEBUG_MODE:
# #     logger.info(f"🔧 设置jieba缓存目录: {jieba_cache_dir}")

# # # ==================== 模型使用统计类 ====================

# # class ModelUsageStats:
# #     def __init__(self):
# #         self.strict_model_name = ""
# #         self.cot_model_name = ""
# #         self.strict_model_calls = 0
# #         self.cot_model_calls = 0
# #         self.lock = Lock()
# #     def increment_strict(self):
# #         with self.lock:
# #             self.strict_model_calls += 1
# #     def increment_cot(self):
# #         with self.lock:
# #             self.cot_model_calls += 1
# #     def get_stats(self) -> Dict[str, Any]:
# #         with self.lock:
# #             return {
# #                 "timestamp": datetime.now().isoformat(),
# #                 "strict_model": self.strict_model_name,
# #                 "cot_model": self.cot_model_name,
# #                 "strict_model_calls": self.strict_model_calls,
# #                 "cot_model_calls": self.cot_model_calls,
# #                 "total_calls": self.strict_model_calls + self.cot_model_calls
# #             }
# #     def print_summary(self):
# #         stats = self.get_stats()
# #         print("\n" + "="*60)
# #         print("📊 模型使用统计摘要")
# #         print("="*60)
# #         print(f"📅 时间: {stats['timestamp']}")
# #         print(f"🤖 严格模型: {stats['strict_model']}")
# #         print(f"🤖 CoT模型: {stats['cot_model']}")
# #         print(f"📞 严格模型调用次数: {stats['strict_model_calls']}")
# #         print(f"📞 CoT模型调用次数: {stats['cot_model_calls']}")
# #         print(f"📞 总调用次数: {stats['total_calls']}")
# #         print("="*60)

# # # ==================== 问题定义 ====================

# GLOBAL_PROBLEMS = {
#     "Logical Mainline Confusion": {
#         "description": "When merging disaster event information, the logical sequence or narrative flow of the original text becomes disrupted or confused, leading to difficulty in understanding the chronological order, causal relationships, or narrative progression of events.",
#         "evaluation_method": "Extract logical mainlines from both original text and fused text, then perform strict comparison to determine if fused text maintains the logical integrity, coherence, and proper sequence of the original."
#     }
# }

# # # 定义逻辑问题分类
# # LOGICAL_PROBLEM_CATEGORIES = {
# #     "time_order_inaccuracy": "时间顺序不精确",
# #     "narrative_incoherence": "叙述不连贯",
# #     "causal_sequence_confusion": "因果顺序错乱",
# #     "other_logical_issues": "其他逻辑问题"
# # }

# # # ==================== Prompt 模板 ====================

# GLOBAL_EVALUATE_PROMPT = """
# As a professional news editor, please evaluate whether the following fused text contains the 'Logical Mainline Confusion' problem.

# ## Problem Description: {problem_description}

# ## IMPORTANT EVALUATION RULES:
# 1. **Focus ONLY on logical problems**: Do NOT include factual omission issues in this evaluation. Factual omissions (missing key facts, details, or information) are NOT considered logical problems.
# 2. **One problem per category**: Each identified logical problem must be assigned to ONLY ONE of the four categories below. Do NOT duplicate the same problem in multiple categories.

# ## STRICT EVALUATION REQUIREMENTS:
# 1. **Extract logical mainlines with high precision**: 
#    - For original text: Identify ALL key events, their exact chronological order, narrative flow transitions, and logical progression with attention to subtle connections.
#    - For fused text: Map the complete logical structure, noting any deviations, omissions, or reordering of events.

# 2. **Perform rigorous comparison and classify problems**: 
#    - Compare the two logical mainlines using strict criteria.
#    - **CRITICAL: Classify any identified LOGICAL problems (NOT factual omissions) into these specific categories**:
#      a. **Time Order Inaccuracy**: Events are not in exact chronological order, temporal markers are inconsistent, or timing is ambiguous.
#      b. **Narrative Incoherence**: The story flow is disrupted, lacks smooth transitions, or jumps between unrelated points.
#      c. **Causal Sequence Confusion**: Cause-effect relationships are unclear, reversed, or missing, making it hard to understand why events happened.
#      d. **Other Logical Issues**: Any other logical problems not covered above (e.g., contradictory information, illogical conclusions).
#    - **Each problem goes to ONLY ONE category** - choose the most appropriate category.
#    - Apply stricter threshold: Even minor logical disruptions that could potentially confuse readers should be flagged.

# 3. **Filter out factual omissions**: 
#    - If a problem is about missing facts, details, or information (factual omission), it should NOT be counted as a logical problem.
#    - Only problems related to logical sequence, narrative flow, or causal relationships should be included.

# 4. **Strict Decision Criteria**: 
#    - If the fused text contains ANY of the following LOGICAL problems (not factual omissions), the problem exists:
#      * Reordered or misplaced events affecting logical sequence
#      * Inconsistent or ambiguous temporal sequencing  
#      * Disrupted or confusing narrative flow
#      * Unclear, reversed, or missing cause-effect relationships
#      * Contradictory information or illogical conclusions
#    - Partial matches or minor logical issues should still be flagged as problems.
#    - Factual omissions should be noted separately but NOT counted as logical problems.

# ## Text Content:
# **Original Text**: {original_text}
# **Fused Text**: {fused_text}

# ## Output Requirements:
# - Answer in the same language as the input text.
# - Provide structured evidence in JSON format ALWAYS, regardless of whether the problem exists.

# ## Output Format: Please return the evaluation in JSON format, including the following fields:
# - "problem_exists": Boolean, indicating whether ANY logical problem exists (be STRICT in judgment)
# - "evidence": Structured analysis with the following format:
#   - "analysis_summary": Brief but detailed summary of the evaluation
#   - "original_logical_mainline": Extraction of logical mainline from original text (list of key steps/events in order)
#   - "fused_logical_mainline": Extraction of logical mainline from fused text (list of key steps/events in order)
#   - "logical_problem_categories": List of logical problem categories detected (select from: "time_order_inaccuracy", "narrative_incoherence", "causal_sequence_confusion", "other_logical_issues")
#   - "categorized_problems": Detailed breakdown of LOGICAL problems by category (factual omissions excluded):
#     - "time_order_inaccuracy": List of specific time order problems (each problem appears only once)
#     - "narrative_incoherence": List of specific narrative coherence problems (each problem appears only once)
#     - "causal_sequence_confusion": List of specific causal sequence problems (each problem appears only once)
#     - "other_logical_issues": List of other logical problems (each problem appears only once)
#   - "factual_omissions_noted": List of factual omission issues found (if any, but these are NOT counted as logical problems)
#   - "severity_assessment": Overall severity of LOGICAL problems only (High/Medium/Low/None)
#   - "verdict": Final verdict on whether logical mainline confusion exists (Yes/No)

# ## Important Notes:
# - BE STRICT in your evaluation. Minor logical issues should be flagged as problems.
# - Each logical problem must be assigned to ONLY ONE category.
# - Factual omission issues (missing facts/information) should be listed separately but NOT counted as logical problems.
# - ALWAYS provide a complete, structured analysis in the evidence field.
# - The goal is to ensure the fused text maintains the logical structure and progression of the original text.
# - Your analysis should be thorough enough to identify even subtle logical disruptions.
# Please begin the strict evaluation:
# """

# # # ==================== 评估系统 ====================

# # class EvaluationSystem:
# #     class SimpleResponse:
# #         def __init__(self, content: str):
# #             self.content = content

# #     def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, gpu_device: str = "cuda:0"):
# #         self.strict_model_config = strict_model_config
# #         self.cot_model_config = cot_model_config if cot_model_config else strict_model_config.copy()
# #         self.strict_model_name = strict_model_config.get("model", "unknown")
# #         self.cot_model_name = self.cot_model_config.get("model", "unknown")
# #         self.retry_limit = API_RETRY_LIMIT
# #         self.gpu_device = gpu_device

# #         self.model_stats = ModelUsageStats()
# #         self.model_stats.strict_model_name = self.strict_model_name
# #         self.model_stats.cot_model_name = self.cot_model_name

# #         if DEBUG_MODE:
# #             logger.info(f"🤖 初始化评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
# #             logger.info(f"🎯 指定GPU设备: {gpu_device}")

# #         self.llm_strict = self._initialize_llm(strict_model_config)
# #         self.llm_cot = self._initialize_llm(self.cot_model_config)

# #     def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
# #         temperature = model_config.get("temperature", 0.3)  # 降低温度以提高确定性
# #         max_tokens = model_config.get("max_tokens", 4000)
# #         model_name = model_config.get("model_name", model_config.get("model", "gpt-4"))
# #         api_key = model_config.get("api_key", "")
# #         base_url = model_config.get("base_url", "https://api.openai.com/v1")

# #         if api_key == "EMPTY" or api_key == "vllm":
# #             api_key = "vllm"
# #             if not base_url or base_url == "https://api.openai.com/v1":
# #                 base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# #             logger.info(f"🔧 使用vllm本地模型: {model_name}, 地址: {base_url}")

# #         if DEBUG_MODE:
# #             logger.info(f"🔧 初始化LLM: 模型={model_name}, 温度={temperature}, 最大令牌={max_tokens}, 基础URL={base_url}")

# #         return ChatOpenAI(
# #             model_name=model_name,
# #             temperature=temperature,
# #             max_tokens=max_tokens,
# #             openai_api_key=api_key,
# #             openai_api_base=base_url,
# #             request_timeout=REQUEST_TIMEOUT,
# #             max_retries=self.retry_limit,
# #         )

# #     def estimate_tokens(self, text: str) -> int:
# #         chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
# #         english_chars = len(re.findall(r'[a-zA-Z]', text))
# #         other_chars = len(text) - chinese_chars - english_chars
# #         estimated_tokens = int(chinese_chars * 1.5 + english_chars * 0.75 + other_chars * 0.5)
# #         return max(estimated_tokens, 1)

# #     # 提取文本的通用方法
# #     def _extract_text_from_llm_response(self, response_raw: Any) -> str:
# #         if response_raw is None:
# #             return ""
# #         if isinstance(response_raw, str):
# #             return response_raw
# #         if hasattr(response_raw, "content") and isinstance(getattr(response_raw, "content"), str):
# #             return getattr(response_raw, "content")
# #         if hasattr(response_raw, "text") and isinstance(getattr(response_raw, "text"), str):
# #             return getattr(response_raw, "text")
# #         # langchain-like generations
# #         try:
# #             gens = getattr(response_raw, "generations", None)
# #             if gens:
# #                 first = gens[0]
# #                 if isinstance(first, list) and len(first) > 0:
# #                     candidate = first[0]
# #                     txt = getattr(candidate, "text", None)
# #                     if isinstance(txt, str):
# #                         return txt
# #                 else:
# #                     candidate = first
# #                     txt = getattr(candidate, "text", None)
# #                     if isinstance(txt, str):
# #                         return txt
# #         except Exception:
# #             pass
# #         # openai-like dict
# #         if isinstance(response_raw, dict):
# #             choices = response_raw.get("choices")
# #             if isinstance(choices, list) and len(choices) > 0:
# #                 first = choices[0]
# #                 if isinstance(first, dict):
# #                     msg = first.get("message")
# #                     if isinstance(msg, dict) and isinstance(msg.get("content"), str):
# #                         return msg.get("content")
# #                     if isinstance(first.get("text"), str):
# #                         return first.get("text")
# #             for k in ("content", "text", "message"):
# #                 if k in response_raw and isinstance(response_raw[k], str):
# #                     return response_raw[k]
# #         # choices attribute fallback
# #         if hasattr(response_raw, "choices"):
# #             try:
# #                 choices = getattr(response_raw, "choices")
# #                 if isinstance(choices, (list, tuple)) and len(choices) > 0:
# #                     first = choices[0]
# #                     if isinstance(first, dict):
# #                         if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
# #                             return first["message"]["content"]
# #                         if "text" in first and isinstance(first["text"], str):
# #                             return first["text"]
# #                     else:
# #                         if hasattr(first, "message") and isinstance(getattr(first, "message"), dict):
# #                             mc = getattr(first, "message").get("content")
# #                             if isinstance(mc, str):
# #                                 return mc
# #                         if hasattr(first, "text"):
# #                             t = getattr(first, "text")
# #                             if isinstance(t, str):
# #                                 return t
# #             except Exception:
# #                 pass
# #         # fallback to str()
# #         try:
# #             return str(response_raw)
# #         except Exception:
# #             return ""

# #     def safe_llm_call(self, messages, llm_instance=None, max_retries=3, base_delay=1, step_name: str = "unknown"):
# #         llm = llm_instance if llm_instance else self.llm_strict
# #         try:
# #             if llm_instance is None or llm_instance == self.llm_strict:
# #                 self.model_stats.increment_strict()
# #             else:
# #                 self.model_stats.increment_cot()
# #         except Exception:
# #             pass

# #         for attempt in range(max_retries + 1):
# #             try:
# #                 response_raw = llm.invoke(messages)
# #                 text = self._extract_text_from_llm_response(response_raw)
# #                 if DEBUG_MODE:
# #                     logger.info(f"📞 模型调用成功: {step_name}, 文本长度: {len(text)}")
# #                 return EvaluationSystem.SimpleResponse(text)
# #             except Exception as e:
# #                 if attempt < max_retries:
# #                     delay = base_delay * (2 ** attempt)
# #                     if DEBUG_MODE:
# #                         logger.warning(f"⚠️ API调用失败，第{attempt + 1}次重试: {e}")
# #                     time.sleep(delay)
# #                 else:
# #                     logger.error(f"❌ LLM API 调用最终失败: {e}")
# #         return None

# #     # 提取 JSON 块（支持嵌套 {} 或 []）
# #     def _extract_json_block(self, text: str) -> Optional[str]:
# #         if not text or not isinstance(text, str):
# #             return None
# #         start_idx = None
# #         open_char = None
# #         for i, ch in enumerate(text):
# #             if ch == '{' or ch == '[':
# #                 start_idx = i
# #                 open_char = ch
# #                 break
# #         if start_idx is None:
# #             return None
# #         close_char = '}' if open_char == '{' else ']'
# #         stack = []
# #         in_string = False
# #         escape = False
# #         for j in range(start_idx, len(text)):
# #             ch = text[j]
# #             if escape:
# #                 escape = False
# #                 continue
# #             if ch == '\\':
# #                 escape = True
# #                 continue
# #             if ch == '"' and not escape:
# #                 in_string = not in_string
# #                 continue
# #             if in_string:
# #                 continue
# #             if ch == open_char:
# #                 stack.append(ch)
# #             elif ch == close_char:
# #                 if not stack:
# #                     return None
# #                 stack.pop()
# #                 if not stack:
# #                     return text[start_idx:j+1]
# #         return None

# #     def _normalize_parsed(self, parsed: Any) -> Dict[str, Any]:
# #         if isinstance(parsed, dict):
# #             return parsed
# #         if isinstance(parsed, list):
# #             for item in parsed:
# #                 if isinstance(item, dict):
# #                     return item
# #             return {"items": parsed}
# #         return {"value": parsed}

# #     def safe_json_parse(self, response_text: Any) -> Dict[str, Any]:
# #         if isinstance(response_text, dict):
# #             return response_text
# #         if hasattr(response_text, "content"):
# #             text = getattr(response_text, "content")
# #         else:
# #             text = response_text
# #         if isinstance(text, str):
# #             s = text.strip()
# #             # 直接尝试 json.loads
# #             try:
# #                 parsed = json.loads(s)
# #                 return self._normalize_parsed(parsed)
# #             except Exception:
# #                 json_block = self._extract_json_block(s)
# #                 if json_block:
# #                     try:
# #                         parsed = json.loads(json_block)
# #                         return self._normalize_parsed(parsed)
# #                     except Exception:
# #                         pass
# #                 json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s, re.DOTALL)
# #                 if json_match:
# #                     try:
# #                         parsed = json.loads(json_match.group(1))
# #                         return self._normalize_parsed(parsed)
# #                     except Exception:
# #                         pass
# #         # 解析失败返回默认dict
# #         return {"problem_exists": False, "evidence": {}}

# #     # ==================== 全局评估 ====================

# #     def evaluate_global_problem(self, problem_type: str, original_text: str, fused_text: str) -> Dict[str, Any]:
# #         try:
# #             problem_info = GLOBAL_PROBLEMS.get(problem_type, {})
# #             description = problem_info.get("description", "")
# #             prompt = GLOBAL_EVALUATE_PROMPT.format(
# #                 problem_type=problem_type,
# #                 problem_description=description,
# #                 original_text=original_text,
# #                 fused_text=fused_text,
# #             )
# #             messages = [
# #                 SystemMessage(content="As a professional news editor, you are responsible for rigorously evaluating the overall quality of text fusion. For the issue of 'Logical Mainline Confusion,' you must strictly extract the logical mainlines from both the original text and the fused text, and conduct a strict comparative analysis. Note: Each identified logical problem must be assigned to only one category; factual omission issues (missing key information) are not considered logical problems and should be listed separately. Even minor issues should be flagged. Regardless of whether problems exist, you must provide complete and detailed structured analysis evidence."),
# #                 HumanMessage(content=prompt),
# #             ]
# #             response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"global_{problem_type}")
# #             if not response:
# #                 return {
# #                     "problem_exists": False,
# #                     "problem_type": problem_type,
# #                     "evidence": {
# #                         "analysis_summary": "评估失败，无法获取模型响应",
# #                         "original_logical_mainline": [],
# #                         "fused_logical_mainline": [],
# #                         "logical_problem_categories": [],
# #                         "categorized_problems": {
# #                             "time_order_inaccuracy": [],
# #                             "narrative_incoherence": [],
# #                             "causal_sequence_confusion": [],
# #                             "other_logical_issues": []
# #                         },
# #                         "factual_omissions_noted": [],
# #                         "severity_assessment": "None",
# #                         "verdict": "No"
# #                     },
# #                     "evaluation_method": "global_assessment"
# #                 }
# #             eval_data = self.safe_json_parse(response.content)
# #             # eval_data 已归一为 dict
# #             problem_exists = eval_data.get("problem_exists", False)
# #             evidence = eval_data.get("evidence", {})
# #             if not evidence:
# #                 evidence = {
# #                     "analysis_summary": "模型未返回结构化证据",
# #                     "original_logical_mainline": [],
# #                     "fused_logical_mainline": [],
# #                     "logical_problem_categories": [],
# #                     "categorized_problems": {
# #                         "time_order_inaccuracy": [],
# #                         "narrative_incoherence": [],
# #                         "causal_sequence_confusion": [],
# #                         "other_logical_issues": []
# #                     },
# #                     "factual_omissions_noted": [],
# #                     "severity_assessment": "None",
# #                     "verdict": "No"
# #                 }
            
# #             # 确保categorized_problems字段存在且格式正确
# #             if "categorized_problems" not in evidence:
# #                 evidence["categorized_problems"] = {
# #                     "time_order_inaccuracy": [],
# #                     "narrative_incoherence": [],
# #                     "causal_sequence_confusion": [],
# #                     "other_logical_issues": []
# #                 }
            
# #             # 确保logical_problem_categories字段存在
# #             if "logical_problem_categories" not in evidence:
# #                 evidence["logical_problem_categories"] = []
            
# #             # 确保factual_omissions_noted字段存在
# #             if "factual_omissions_noted" not in evidence:
# #                 evidence["factual_omissions_noted"] = []
            
#             return {
#                 "problem_exists": bool(problem_exists),
#                 "problem_type": problem_type,
#                 "evidence": evidence,
#                 "evaluation_method": "global_assessment"
#             }
#         except Exception as e:
#             logger.error(f"❌ 整体评估失败 {problem_type}: {e}")
#             return {
#                 "problem_exists": False,
#                 "problem_type": problem_type,
#                 "evidence": {
#                     "analysis_summary": f"评估异常: {str(e)}",
#                     "original_logical_mainline": [],
#                     "fused_logical_mainline": [],
#                     "logical_problem_categories": [],
#                     "categorized_problems": {
#                         "time_order_inaccuracy": [],
#                         "narrative_incoherence": [],
#                         "causal_sequence_confusion": [],
#                         "other_logical_issues": []
#                     },
#                     "factual_omissions_noted": [],
#                     "severity_assessment": "None",
#                     "verdict": "No"
#                 },
#                 "evaluation_method": "global_assessment"
#             }

#     # ==================== 主评估流程 ====================

#     def evaluate(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
#         logger.info("🔄 开始评估（整体文本逻辑主线评估）")
#         original_text = news_item.get("input_text", "")
#         fused_text = news_item.get("fused_content", "")
        
#         if not original_text.strip() or not fused_text.strip():
#             logger.warning(f"❌ 新闻内容为空，无法评估")
#             return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
        
#         global_results = self.global_evaluation_stage(original_text, fused_text)
#         global_classification = self._classify_global_problems(global_results["global_errors"])
        
#         # 汇总问题分类
#         problem_category_summary = self._summarize_problem_categories(global_results["global_assessments"])
        
#         # 汇总事实遗漏统计
#         factual_omissions_summary = self._summarize_factual_omissions(global_results["global_assessments"])
        
#         result = {
#             "evaluation_success": True,
#             "global_evaluation": global_results,
#             "problem_classification": {
#                 "global_problems": global_classification,
#                 "logical_problem_category_summary": problem_category_summary,
#                 "factual_omissions_summary": factual_omissions_summary,
#                 "total_problems": global_results["total_global_problems"]
#             },
#             "model_usage_stats": self.model_stats.get_stats()
#         }
        
#         logger.info("✅ 评估完成")
#         return result

#     def global_evaluation_stage(self, original_text: str, fused_text: str) -> Dict[str, Any]:
#         logger.info("🌐 整体文本逻辑主线评估")
#         if not original_text.strip() or not fused_text.strip():
#             return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        
#         if not GLOBAL_PROBLEMS:
#             logger.info("没有启用的整体问题类型，跳过整体评估")
#             return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        
#         global_assessments = []
#         global_errors = []
        
#         with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(GLOBAL_PROBLEMS))) as executor:
#             future_to_problem = {
#                 executor.submit(self.evaluate_global_problem, problem_type, original_text, fused_text): problem_type
#                 for problem_type in GLOBAL_PROBLEMS.keys()
#             }
            
#             for future in as_completed(future_to_problem):
#                 problem_type = future_to_problem[future]
#                 try:
#                     result = future.result()
#                     global_assessments.append({
#                         "problem_type": result["problem_type"],
#                         "problem_exists": result.get("problem_exists", False),
#                         "evidence": result.get("evidence", {}),
#                         "evaluation_method": result.get("evaluation_method", ""),
#                     })
                    
#                     if result.get("problem_exists", False):
#                         global_errors.append({
#                             "error_type": result["problem_type"],
#                             "evidence": result.get("evidence", {}),
#                             "original_text": original_text,
#                             "fused_text": fused_text,
#                         })
                        
#                 except Exception as e:
#                     logger.error(f"❌ 整体问题类型评估异常: {problem_type}, {e}")
#                     global_assessments.append({
#                         "problem_type": problem_type,
#                         "problem_exists": False,
#                         "evidence": {
#                             "analysis_summary": f"评估异常: {str(e)}",
#                             "original_logical_mainline": [],
#                             "fused_logical_mainline": [],
#                             "logical_problem_categories": [],
#                             "categorized_problems": {
#                                 "time_order_inaccuracy": [],
#                                 "narrative_incoherence": [],
#                                 "causal_sequence_confusion": [],
#                                 "other_logical_issues": []
#                             },
#                             "factual_omissions_noted": [],
#                             "severity_assessment": "None",
#                             "verdict": "No"
#                         },
#                         "evaluation_method": "global_assessment",
#                         "error": str(e)
#                     })
        
#         return {
#             "global_assessments": global_assessments,
#             "global_errors": global_errors,
#             "total_global_problems": len(global_errors)
#         }

#     def _classify_global_problems(self, global_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
#         classification = defaultdict(int)
#         for error in global_errors:
#             error_type = error.get("error_type", "未知类型")
#             classification[error_type] += 1
#         return {"by_type": dict(classification), "total_count": len(global_errors)}
    
#     def _summarize_problem_categories(self, global_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """汇总逻辑问题分类统计"""
#         category_counts = defaultdict(int)
#         all_problems = []
        
#         for assessment in global_assessments:
#             if assessment.get("problem_exists", False):
#                 evidence = assessment.get("evidence", {})
#                 logical_categories = evidence.get("logical_problem_categories", [])
                
#                 for category in logical_categories:
#                     category_counts[category] += 1
                
#                 # 收集具体问题
#                 categorized_problems = evidence.get("categorized_problems", {})
#                 for category, problems in categorized_problems.items():
#                     if problems:
#                         all_problems.extend([{
#                             "category": category,
#                             "problem": problem
#                         } for problem in problems])
        
#         # 转换为中文显示
#         category_display = {}
#         for eng_category, chn_name in LOGICAL_PROBLEM_CATEGORIES.items():
#             if eng_category in category_counts:
#                 category_display[chn_name] = category_counts[eng_category]
        
#         return {
#             "category_counts": dict(category_counts),
#             "category_display": category_display,
#             "total_categories": len(category_counts),
#             "all_problems": all_problems
#         }
    
#     def _summarize_factual_omissions(self, global_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """汇总事实遗漏统计"""
#         all_factual_omissions = []
        
#         for assessment in global_assessments:
#             evidence = assessment.get("evidence", {})
#             factual_omissions = evidence.get("factual_omissions_noted", [])
            
#             if factual_omissions:
#                 all_factual_omissions.extend(factual_omissions)
        
#         return {
#             "factual_omissions_count": len(all_factual_omissions),
#             "factual_omissions_list": all_factual_omissions,
#             "has_factual_omissions": len(all_factual_omissions) > 0
#         }

#     def get_model_usage_stats(self):
#         return self.model_stats.get_stats()

#     def print_model_usage_summary(self):
#         self.model_stats.print_summary()

# # ==================== 处理器与工具函数 ====================

# def create_evaluator(strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
#     if strict_model_name is None:
#         strict_model_name = DEFAULT_TEACHER_MODEL
#     if cot_model_name is None:
#         cot_model_name = strict_model_name
#     strict_model_config = get_model_deployment_config(strict_model_name)
#     cot_model_config = get_model_deployment_config(cot_model_name)
#     return EvaluationSystem(strict_model_config, cot_model_config, gpu_device=gpu_device)

# class EvaluationProcessor:
#     def __init__(self, strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
#         self.evaluator = create_evaluator(strict_model_name, cot_model_name, gpu_device=gpu_device)
#         self.max_workers = max(1, MAX_WORKERS // 2)
#         self.batch_size = max(1, BATCH_SIZE // 2)
#         self.strict_model_name = strict_model_name or DEFAULT_TEACHER_MODEL
#         self.cot_model_name = cot_model_name or self.strict_model_name

#     def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             evaluation_result = self.evaluator.evaluate(news_item)
#             return self._create_evaluated_news(news_item, evaluation_result)
#         except Exception as e:
#             logger.error(f"❌ 评估失败: {e}")
#             return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

#     def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
#         evaluated_news = original_news.copy()
#         evaluated_news.update({
#             "evaluation": evaluation_result,
#             "evaluation_success": evaluation_result.get("evaluation_success", False),
#         })
        
#         if evaluation_result.get("evaluation_success", False):
#             evaluated_news.update({
#                 "global_problem_count": evaluation_result["global_evaluation"]["total_global_problems"],
#                 "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
#             })
        
#         return evaluated_news

#     def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         logger.info(f"开始批量评估 {len(news_list)} 篇新闻")
#         processed_news = []
#         success_count = 0
#         total_global_problems = 0
#         batch_size = min(self.batch_size, 2)
        
#         for batch_start in range(0, len(news_list), batch_size):
#             batch_end = min(batch_start + batch_size, len(news_list))
#             batch = news_list[batch_start:batch_end]
#             logger.info(f"🔄 评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
            
#             batch_results = []
#             pbar = tqdm(total=len(batch), desc=f"评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
            
#             with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
#                 future_to_news = {executor.submit(self.process_single_news, news): news for news in batch}
                
#                 for future in as_completed(future_to_news):
#                     news_item = future_to_news[future]
#                     try:
#                         result = future.result()
#                         batch_results.append(result)
                        
#                         if result.get("evaluation_success", False):
#                             success_count += 1
#                             total_global_problems += result.get("global_problem_count", 0)
                            
#                     except Exception as e:
#                         logger.error(f"❌ 评估批次处理失败: {e}")
#                         error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
#                         batch_results.append(error_news)
#                     finally:
#                         if pbar:
#                             pbar.update(1)
            
#             if pbar:
#                 pbar.close()
            
#             processed_news.extend(batch_results)
#             batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
#             logger.info(f"✅ 评估批次完成: 成功 {batch_success}/{len(batch_results)}")
        
#         self.evaluator.print_model_usage_summary()
#         logger.info(f"🎉 批量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
#         logger.info(f"📊 总逻辑主线问题: {total_global_problems}")
        
#         return processed_news

# # ==================== 其他工具函数 ====================

# def load_news_for_evaluation(file_path: str) -> List[Dict[str, Any]]:
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
        
#         if isinstance(data, list):
#             logger.info(f"加载 {len(data)} 条新闻（旧格式）")
#             return data
#         elif isinstance(data, dict):
#             if "data" in data and isinstance(data["data"], list):
#                 logger.info(f"加载到嵌套结构，顶层data包含 {len(data['data'])} 条记录")
#                 if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
#                     first_item = data["data"][0]
#                     if any(key in first_item for key in ["fused_content", "input_text", "alignment"]):
#                         logger.info(f"直接使用顶层data中的 {len(data['data'])} 条新闻项")
#                         return data["data"]
#                     elif "data" in first_item and isinstance(first_item["data"], list):
#                         inner_items = []
#                         for outer_item in data["data"]:
#                             if isinstance(outer_item, dict) and "data" in outer_item and isinstance(outer_item["data"], list):
#                                 inner_items.extend(outer_item["data"])
#                         if inner_items:
#                             logger.info(f"从嵌套结构中提取 {len(inner_items)} 条内层新闻项")
#                             return inner_items
#             logger.warning("输入JSON是对象，包装成列表")
#             return [data]
#         else:
#             logger.error("不支持的输入JSON根类型")
#             return []
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
#     new_base_name = f"{file_name_without_ext}_logic_{model_suffix}{file_ext}"
#     return os.path.join(dir_name, new_base_name)

# def save_evaluated_news(news_list: List[Dict[str, Any]], file_path: str, stats: Optional[Dict[str, Any]] = None, strict_model_name: str = "unknown", cot_model_name: str = "unknown"):
#     try:
#         os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
#         output_data = {
#             "evaluation_stats": stats if stats else {},
#             "strict_model": strict_model_name,
#             "cot_model": cot_model_name,
#             "total_items": len(news_list),
#             "data": news_list
#         }
#         with open(file_path, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, ensure_ascii=False, indent=2)
#         logger.info(f"💾 成功保存 {len(news_list)} 篇评估新闻到 {file_path}")
#     except Exception as e:
#         logger.error(f"❌ 保存文件失败: {e}")

# def generate_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
#     if not successful_evaluations:
#         return {
#             "summary": {
#                 "total_news_evaluated": 0,
#                 "average_global_problems": 0.0,
#                 "average_total_problems": 0.0,
#             },
#             "global_problems_classification": {"by_type": {}, "total_count": 0},
#             "logical_problem_category_summary": {},
#             "factual_omissions_summary": {}
#         }
    
#     total_news = len(successful_evaluations)
#     global_problems_by_type = defaultdict(int)
#     category_totals = defaultdict(int)
#     factual_omissions_total = 0
    
#     for news in successful_evaluations:
#         evaluation = news.get("evaluation", {})
#         problem_classification = evaluation.get("problem_classification", {})
        
#         # 统计问题类型
#         global_problems = problem_classification.get("global_problems", {})
#         for problem_type, count in global_problems.get("by_type", {}).items():
#             global_problems_by_type[problem_type] += count
        
#         # 统计逻辑问题分类
#         logical_summary = problem_classification.get("logical_problem_category_summary", {})
#         category_counts = logical_summary.get("category_counts", {})
#         for category, count in category_counts.items():
#             category_totals[category] += count
        
#         # 统计事实遗漏
#         factual_summary = problem_classification.get("factual_omissions_summary", {})
#         factual_omissions_total += factual_summary.get("factual_omissions_count", 0)
    
#     avg_global_problems = sum(news.get("global_problem_count", 0) for news in successful_evaluations) / total_news
#     avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
    
#     # 转换为中文显示
#     category_display = {}
#     for eng_category, chn_name in LOGICAL_PROBLEM_CATEGORIES.items():
#         if eng_category in category_totals:
#             category_display[chn_name] = category_totals[eng_category]
    
#     return {
#         "summary": {
#             "total_news_evaluated": total_news,
#             "average_global_problems": round(avg_global_problems, 2),
#             "average_total_problems": round(avg_total_problems, 2),
#         },
#         "global_problems_classification": {
#             "by_type": dict(global_problems_by_type),
#             "total_count": sum(global_problems_by_type.values())
#         },
#         "logical_problem_category_summary": {
#             "category_totals": dict(category_totals),
#             "category_display": category_display,
#             "total_categories": len(category_totals)
#         },
#         "factual_omissions_summary": {
#             "total_factual_omissions": factual_omissions_total,
#             "average_factual_omissions": factual_omissions_total / total_news if total_news > 0 else 0
#         }
#     }

# # ==================== 主函数 ====================

# def main():
#     parser = argparse.ArgumentParser(description="新闻逻辑主线评估工具")
#     parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
#     parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
#     parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
#     parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
#     parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
#     parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
#     parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子，默认 {DEFAULT_SEED}")
    
#     args = parser.parse_args()

#     if not os.path.exists(args.input):
#         print(f"❌ 输入文件不存在: {args.input}")
#         sys.exit(1)

#     # 设置用户指定的随机种子
#     if args.seed != DEFAULT_SEED:
#         print(f"🔧 使用用户指定的随机种子: {args.seed}")
#         set_global_seed(args.seed)

#     print("📥 加载新闻数据...")
#     news_data = load_news_for_evaluation(args.input)
#     if not news_data:
#         print("❌ 没有加载到新闻数据")
#         sys.exit(1)

#     if args.sample > 0:
#         news_data = news_data[:args.sample]
#         print(f"🔬 采样处理前 {args.sample} 篇新闻")

#     print("🚀 初始化评估处理器...")
#     print(f"🎯 使用GPU设备: {args.gpu_device}")
#     processor = EvaluationProcessor(
#         strict_model_name=args.strict_model,
#         cot_model_name=args.cot_model,
#         gpu_device=args.gpu_device
#     )

#     print("🔄 开始逻辑主线评估...")
#     evaluated_news = processor.process_news_batch(news_data)

#     stats = processor.evaluator.get_model_usage_stats()

#     strict_model_name = processor.strict_model_name
#     cot_model_name = processor.cot_model_name
#     output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)

#     print("💾 保存结果...")
#     save_evaluated_news(evaluated_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)

#     summary = generate_summary_report(evaluated_news)
#     print("\n📊 评估摘要:")
#     print(f" 评估新闻总数: {summary['summary']['total_news_evaluated']}")
#     print(f" 平均逻辑主线问题数: {summary['summary']['average_global_problems']}")
#     print(f" 平均总问题数: {summary['summary']['average_total_problems']}")

#     if summary['global_problems_classification']['total_count'] > 0:
#         print(f" 逻辑主线问题分类: {summary['global_problems_classification']['by_type']}")
    
#     if summary['logical_problem_category_summary']['total_categories'] > 0:
#         print("\n🔍 逻辑问题分类统计:")
#         for chn_name, count in summary['logical_problem_category_summary']['category_display'].items():
#             print(f"  {chn_name}: {count}")
    
#     if summary['factual_omissions_summary']['total_factual_omissions'] > 0:
#         print(f"\n📝 事实遗漏统计: {summary['factual_omissions_summary']['total_factual_omissions']} 处")
#         print(f"  平均每篇新闻事实遗漏: {summary['factual_omissions_summary']['average_factual_omissions']:.2f}")
    
#     # 输出随机种子信息
#     print(f"\n🌱 评估使用的随机种子: {args.seed}")

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
import torch

# 导入配置（请确保你的 config.py 可用并提供所需变量/函数）
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

# ==================== 设置全局随机种子以确保可复现性 ====================
def set_global_seed(seed: int = 42):
    """设置所有可能的随机种子以确保评估的可复现性"""
    print(f"🔧 设置全局随机种子: {seed}")
    
    # 设置Python内置随机种子
    random.seed(seed)
    
    # 设置numpy随机种子
    np.random.seed(seed)
    
    # 设置torch随机种子（如果可用）
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ 全局随机种子已设置为 {seed}")

# 设置默认随机种子（可在命令行参数中覆盖）
DEFAULT_SEED = 42
set_global_seed(DEFAULT_SEED)

# 日志配置
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# 修复jieba缓存目录问题
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
                "total_calls": self.strict_model_calls + self.cot_model_calls
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
        print("="*60)

# ==================== 问题定义 ====================

GLOBAL_PROBLEMS = {
    "Logical Mainline Confusion": {
        "description": "When merging disaster event information, the logical sequence or narrative flow of the original text becomes disrupted or confused, leading to difficulty in understanding the chronological order, causal relationships, or narrative progression of events.",
        "evaluation_method": "Extract logical mainlines from both original text and fused text, then perform strict comparison to determine if fused text maintains the logical integrity, coherence, and proper sequence of the original."
    }
}

# 定义逻辑问题分类
LOGICAL_PROBLEM_CATEGORIES = {
    "time_order_inaccuracy": "时间顺序不精确",
    "narrative_incoherence": "叙述不连贯",
    "causal_sequence_confusion": "因果顺序错乱",
    "other_logical_issues": "其他逻辑问题"
}

# ==================== Prompt 模板 ====================

GLOBAL_EVALUATE_PROMPT = """
As a professional news editor, please evaluate whether the following fused text contains the 'Logical Mainline Confusion' problem.

## Problem Description: {problem_description}

## IMPORTANT EVALUATION RULES:
1. **Focus ONLY on logical problems**: Do NOT include factual omission issues in this evaluation. Factual omissions (missing key facts, details, or information) are NOT considered logical problems.
2. **One problem per category**: Each identified logical problem must be assigned to ONLY ONE of the four categories below. Do NOT duplicate the same problem in multiple categories.

## STRICT EVALUATION REQUIREMENTS:
1. **Extract logical mainlines with high precision**: 
   - For original text: Identify ALL key events, their exact chronological order, narrative flow transitions, and logical progression with attention to subtle connections.
   - For fused text: Map the complete logical structure, noting any deviations, omissions, or reordering of events.

2. **Perform rigorous comparison and classify problems**: 
   - Compare the two logical mainlines using strict criteria.
   - **CRITICAL: Classify any identified LOGICAL problems (NOT factual omissions) into these specific categories**:
     a. **Time Order Inaccuracy**: Events are not in exact chronological order, temporal markers are inconsistent, or timing is ambiguous.
     b. **Narrative Incoherence**: The story flow is disrupted, lacks smooth transitions, or jumps between unrelated points.
     c. **Causal Sequence Confusion**: Cause-effect relationships are unclear, reversed, or missing, making it hard to understand why events happened.
     d. **Other Logical Issues**: Any other logical problems not covered above (e.g., contradictory information, illogical conclusions).
   - **Each problem goes to ONLY ONE category** - choose the most appropriate category.
   - Apply stricter threshold: Even minor logical disruptions that could potentially confuse readers should be flagged.

3. **Filter out factual omissions**: 
   - If a problem is about missing facts, details, or information (factual omission), it should NOT be counted as a logical problem.
   - Only problems related to logical sequence, narrative flow, or causal relationships should be included.

4. **Strict Decision Criteria**: 
   - If the fused text contains ANY of the following LOGICAL problems (not factual omissions), the problem exists:
     * Reordered or misplaced events affecting logical sequence
     * Inconsistent or ambiguous temporal sequencing  
     * Disrupted or confusing narrative flow
     * Unclear, reversed, or missing cause-effect relationships
     * Contradictory information or illogical conclusions
   - Partial matches or minor logical issues should still be flagged as problems.
   - Factual omissions should be noted separately but NOT counted as logical problems.

## Text Content:
**Original Text**: {original_text}
**Fused Text**: {fused_text}

## Output Requirements:
- Answer in the same language as the input text.
- Provide structured evidence in JSON format ALWAYS, regardless of whether the problem exists.

## Output Format: Please return the evaluation in JSON format, including the following fields:
- "problem_exists": Boolean, indicating whether ANY logical problem exists (be STRICT in judgment)
- "evidence": Structured analysis with the following format:
  - "analysis_summary": Brief but detailed summary of the evaluation
  - "original_logical_mainline": Extraction of logical mainline from original text (list of key steps/events in order)
  - "fused_logical_mainline": Extraction of logical mainline from fused text (list of key steps/events in order)
  - "logical_problem_categories": List of logical problem categories detected (select from: "time_order_inaccuracy", "narrative_incoherence", "causal_sequence_confusion", "other_logical_issues")
  - "categorized_problems": Detailed breakdown of LOGICAL problems by category (factual omissions excluded):
    - "time_order_inaccuracy": List of specific time order problems (each problem appears only once)
    - "narrative_incoherence": List of specific narrative coherence problems (each problem appears only once)
    - "causal_sequence_confusion": List of specific causal sequence problems (each problem appears only once)
    - "other_logical_issues": List of other logical problems (each problem appears only once)
  - "factual_omissions_noted": List of factual omission issues found (if any, but these are NOT counted as logical problems)
  - "severity_assessment": Overall severity of LOGICAL problems only (High/Medium/Low/None)
  - "verdict": Final verdict on whether logical mainline confusion exists (Yes/No)

## Important Notes:
- BE STRICT in your evaluation. Minor logical issues should be flagged as problems.
- Each logical problem must be assigned to ONLY ONE category.
- Factual omission issues (missing facts/information) should be listed separately but NOT counted as logical problems.
- ALWAYS provide a complete, structured analysis in the evidence field.
- The goal is to ensure the fused text maintains the logical structure and progression of the original text.
- Your analysis should be thorough enough to identify even subtle logical disruptions.
Please begin the strict evaluation:
"""

# ==================== 评估系统 ====================

class EvaluationSystem:
    class SimpleResponse:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, gpu_device: str = "cuda:0"):
        self.strict_model_config = strict_model_config
        self.cot_model_config = cot_model_config if cot_model_config else strict_model_config.copy()
        self.strict_model_name = strict_model_config.get("model", "unknown")
        self.cot_model_name = self.cot_model_config.get("model", "unknown")
        self.retry_limit = API_RETRY_LIMIT
        self.gpu_device = gpu_device

        self.model_stats = ModelUsageStats()
        self.model_stats.strict_model_name = self.strict_model_name
        self.model_stats.cot_model_name = self.cot_model_name

        if DEBUG_MODE:
            logger.info(f"🤖 初始化评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
            logger.info(f"🎯 指定GPU设备: {gpu_device}")

        self.llm_strict = self._initialize_llm(strict_model_config)
        self.llm_cot = self._initialize_llm(self.cot_model_config)

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.3)  # 降低温度以提高确定性
        # 修改 max_tokens 为 12288
        max_tokens = 12288
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")

        if api_key == "EMPTY" or api_key == "vllm":
            api_key = "vllm"
            if not base_url or base_url == "https://api.openai.com/v1":
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            logger.info(f"🔧 使用vllm本地模型: {model_name}, 地址: {base_url}")

        if DEBUG_MODE:
            logger.info(f"🔧 初始化LLM: 模型={model_name}, 温度={temperature}, 最大令牌={max_tokens}, 基础URL={base_url}")

        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT,
            max_retries=self.retry_limit,
        )

    def estimate_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        estimated_tokens = int(chinese_chars * 1.5 + english_chars * 0.75 + other_chars * 0.5)
        return max(estimated_tokens, 1)

    # 提取文本的通用方法
    def _extract_text_from_llm_response(self, response_raw: Any) -> str:
        if response_raw is None:
            return ""
        if isinstance(response_raw, str):
            return response_raw
        if hasattr(response_raw, "content") and isinstance(getattr(response_raw, "content"), str):
            return getattr(response_raw, "content")
        if hasattr(response_raw, "text") and isinstance(getattr(response_raw, "text"), str):
            return getattr(response_raw, "text")
        # langchain-like generations
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
        # openai-like dict
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
        # choices attribute fallback
        if hasattr(response_raw, "choices"):
            try:
                choices = getattr(response_raw, "choices")
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                            return first["message"]["content"]
                        if "text" in first and isinstance(first["text"], str):
                            return first["text"]
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
        # fallback to str()
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

        for attempt in range(max_retries + 1):
            try:
                response_raw = llm.invoke(messages)
                text = self._extract_text_from_llm_response(response_raw)
                if DEBUG_MODE:
                    logger.info(f"📞 模型调用成功: {step_name}, 文本长度: {len(text)}")
                return EvaluationSystem.SimpleResponse(text)
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    if DEBUG_MODE:
                        logger.warning(f"⚠️ API调用失败，第{attempt + 1}次重试: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ LLM API 调用最终失败: {e}")
        return None

    # 提取 JSON 块（支持嵌套 {} 或 []）
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
            # 直接尝试 json.loads
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
        # 解析失败返回默认dict
        return {"problem_exists": False, "evidence": {}}

    # ==================== 全局评估 ====================

    def evaluate_global_problem(self, problem_type: str, original_text: str, fused_text: str) -> Dict[str, Any]:
        try:
            problem_info = GLOBAL_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            prompt = GLOBAL_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                original_text=original_text,
                fused_text=fused_text,
            )
            messages = [
                SystemMessage(content="As a professional news editor, you are responsible for strictly evaluating the overall quality of text fusion. For the 'Logical Mainline Confusion' issue, you must strictly extract the logical mainlines from both the original text and the fused text, and conduct a rigorous comparative analysis. Note: Each identified logical problem must be assigned to only one category; factual omission issues (missing key information) are not considered logical problems and should be listed separately. Even minor issues should be flagged. Regardless of whether a problem exists, you must provide complete and detailed structured analytical evidence."),
                HumanMessage(content=prompt),
            ]
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"global_{problem_type}")
            if not response:
                return {
                    "problem_exists": False,
                    "problem_type": problem_type,
                    "evidence": {
                        "analysis_summary": "评估失败，无法获取模型响应",
                        "original_logical_mainline": [],
                        "fused_logical_mainline": [],
                        "logical_problem_categories": [],
                        "categorized_problems": {
                            "time_order_inaccuracy": [],
                            "narrative_incoherence": [],
                            "causal_sequence_confusion": [],
                            "other_logical_issues": []
                        },
                        "factual_omissions_noted": [],
                        "severity_assessment": "None",
                        "verdict": "No"
                    },
                    "evaluation_method": "global_assessment"
                }
            eval_data = self.safe_json_parse(response.content)
            # eval_data 已归一为 dict
            problem_exists = eval_data.get("problem_exists", False)
            evidence = eval_data.get("evidence", {})
            if not evidence:
                evidence = {
                    "analysis_summary": "模型未返回结构化证据",
                    "original_logical_mainline": [],
                    "fused_logical_mainline": [],
                    "logical_problem_categories": [],
                    "categorized_problems": {
                        "time_order_inaccuracy": [],
                        "narrative_incoherence": [],
                        "causal_sequence_confusion": [],
                        "other_logical_issues": []
                    },
                    "factual_omissions_noted": [],
                    "severity_assessment": "None",
                    "verdict": "No"
                }
            
            # 确保categorized_problems字段存在且格式正确
            if "categorized_problems" not in evidence:
                evidence["categorized_problems"] = {
                    "time_order_inaccuracy": [],
                    "narrative_incoherence": [],
                    "causal_sequence_confusion": [],
                    "other_logical_issues": []
                }
            
            # 确保logical_problem_categories字段存在
            if "logical_problem_categories" not in evidence:
                evidence["logical_problem_categories"] = []
            
            # 确保factual_omissions_noted字段存在
            if "factual_omissions_noted" not in evidence:
                evidence["factual_omissions_noted"] = []
            
            return {
                "problem_exists": bool(problem_exists),
                "problem_type": problem_type,
                "evidence": evidence,
                "evaluation_method": "global_assessment"
            }
        except Exception as e:
            logger.error(f"❌ 整体评估失败 {problem_type}: {e}")
            return {
                "problem_exists": False,
                "problem_type": problem_type,
                "evidence": {
                    "analysis_summary": f"评估异常: {str(e)}",
                    "original_logical_mainline": [],
                    "fused_logical_mainline": [],
                    "logical_problem_categories": [],
                    "categorized_problems": {
                        "time_order_inaccuracy": [],
                        "narrative_incoherence": [],
                        "causal_sequence_confusion": [],
                        "other_logical_issues": []
                    },
                    "factual_omissions_noted": [],
                    "severity_assessment": "None",
                    "verdict": "No"
                },
                "evaluation_method": "global_assessment"
            }

    # ==================== 主评估流程 ====================

    def evaluate(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔄 开始评估（整体文本逻辑主线评估）")
        original_text = news_item.get("input_text", "")
        fused_text = news_item.get("fused_content", "")
        
        if not original_text.strip() or not fused_text.strip():
            logger.warning(f"❌ 新闻内容为空，无法评估")
            return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
        
        global_results = self.global_evaluation_stage(original_text, fused_text)
        global_classification = self._classify_global_problems(global_results["global_errors"])
        
        # 汇总问题分类
        problem_category_summary = self._summarize_problem_categories(global_results["global_assessments"])
        
        # 汇总事实遗漏统计
        factual_omissions_summary = self._summarize_factual_omissions(global_results["global_assessments"])
        
        result = {
            "evaluation_success": True,
            "global_evaluation": global_results,
            "problem_classification": {
                "global_problems": global_classification,
                "logical_problem_category_summary": problem_category_summary,
                "factual_omissions_summary": factual_omissions_summary,
                "total_problems": global_results["total_global_problems"]
            },
            "model_usage_stats": self.model_stats.get_stats()
        }
        
        logger.info("✅ 评估完成")
        return result

    def global_evaluation_stage(self, original_text: str, fused_text: str) -> Dict[str, Any]:
        logger.info("🌐 整体文本逻辑主线评估")
        if not original_text.strip() or not fused_text.strip():
            return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        
        if not GLOBAL_PROBLEMS:
            logger.info("没有启用的整体问题类型，跳过整体评估")
            return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        
        global_assessments = []
        global_errors = []
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(GLOBAL_PROBLEMS))) as executor:
            future_to_problem = {
                executor.submit(self.evaluate_global_problem, problem_type, original_text, fused_text): problem_type
                for problem_type in GLOBAL_PROBLEMS.keys()
            }
            
            for future in as_completed(future_to_problem):
                problem_type = future_to_problem[future]
                try:
                    result = future.result()
                    global_assessments.append({
                        "problem_type": result["problem_type"],
                        "problem_exists": result.get("problem_exists", False),
                        "evidence": result.get("evidence", {}),
                        "evaluation_method": result.get("evaluation_method", ""),
                    })
                    
                    if result.get("problem_exists", False):
                        global_errors.append({
                            "error_type": result["problem_type"],
                            "evidence": result.get("evidence", {}),
                            "original_text": original_text,
                            "fused_text": fused_text,
                        })
                        
                except Exception as e:
                    logger.error(f"❌ 整体问题类型评估异常: {problem_type}, {e}")
                    global_assessments.append({
                        "problem_type": problem_type,
                        "problem_exists": False,
                        "evidence": {
                            "analysis_summary": f"评估异常: {str(e)}",
                            "original_logical_mainline": [],
                            "fused_logical_mainline": [],
                            "logical_problem_categories": [],
                            "categorized_problems": {
                                "time_order_inaccuracy": [],
                                "narrative_incoherence": [],
                                "causal_sequence_confusion": [],
                                "other_logical_issues": []
                            },
                            "factual_omissions_noted": [],
                            "severity_assessment": "None",
                            "verdict": "No"
                        },
                        "evaluation_method": "global_assessment",
                        "error": str(e)
                    })
        
        return {
            "global_assessments": global_assessments,
            "global_errors": global_errors,
            "total_global_problems": len(global_errors)
        }

    def _classify_global_problems(self, global_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        classification = defaultdict(int)
        for error in global_errors:
            error_type = error.get("error_type", "未知类型")
            classification[error_type] += 1
        return {"by_type": dict(classification), "total_count": len(global_errors)}
    
    def _summarize_problem_categories(self, global_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总逻辑问题分类统计"""
        category_counts = defaultdict(int)
        all_problems = []
        
        for assessment in global_assessments:
            if assessment.get("problem_exists", False):
                evidence = assessment.get("evidence", {})
                logical_categories = evidence.get("logical_problem_categories", [])
                
                for category in logical_categories:
                    category_counts[category] += 1
                
                # 收集具体问题
                categorized_problems = evidence.get("categorized_problems", {})
                for category, problems in categorized_problems.items():
                    if problems:
                        all_problems.extend([{
                            "category": category,
                            "problem": problem
                        } for problem in problems])
        
        # 转换为中文显示
        category_display = {}
        for eng_category, chn_name in LOGICAL_PROBLEM_CATEGORIES.items():
            if eng_category in category_counts:
                category_display[chn_name] = category_counts[eng_category]
        
        return {
            "category_counts": dict(category_counts),
            "category_display": category_display,
            "total_categories": len(category_counts),
            "all_problems": all_problems
        }
    
    def _summarize_factual_omissions(self, global_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总事实遗漏统计"""
        all_factual_omissions = []
        
        for assessment in global_assessments:
            evidence = assessment.get("evidence", {})
            factual_omissions = evidence.get("factual_omissions_noted", [])
            
            if factual_omissions:
                all_factual_omissions.extend(factual_omissions)
        
        return {
            "factual_omissions_count": len(all_factual_omissions),
            "factual_omissions_list": all_factual_omissions,
            "has_factual_omissions": len(all_factual_omissions) > 0
        }

    # ==================== 修复缺失结构化输出的方法 ====================
    def _is_missing_structure(self, assessment: Dict[str, Any]) -> bool:
        """判定某一项评估是否因未返回结构化输出而失败"""
        evidence = assessment.get("evidence", {})
        if not evidence:
            return True
        summary = evidence.get("analysis_summary", "")
        if not isinstance(summary, str):
            return True
        # 检查常见失败标记
        if "模型未返回结构化证据" in summary or "评估失败" in summary or "评估异常" in summary:
            return True
        # 检查关键字段是否存在
        required_keys = ["original_logical_mainline", "fused_logical_mainline", 
                         "logical_problem_categories", "categorized_problems", 
                         "factual_omissions_noted", "severity_assessment", "verdict"]
        for key in required_keys:
            if key not in evidence:
                return True
        # 检查 categorized_problems 是否包含四个子字段
        categorized = evidence.get("categorized_problems", {})
        required_subkeys = ["time_order_inaccuracy", "narrative_incoherence", 
                            "causal_sequence_confusion", "other_logical_issues"]
        for subkey in required_subkeys:
            if subkey not in categorized:
                return True
        return False

    def fix_missing_evaluations(self, news_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        """识别并仅对缺失结构化输出的部分发起重评请求"""
        before_missing_count = 0
        
        # 统计修复前的未结构化数量
        for news in news_list:
            evaluation = news.get("evaluation", {})
            for assess in evaluation.get("global_evaluation", {}).get("global_assessments", []):
                if self._is_missing_structure(assess):
                    before_missing_count += 1

        if before_missing_count == 0:
            logger.info("🎉 没有发现未返回结构化输出的评估项，无需修复。")
            return news_list, 0, 0

        logger.info(f"🔧 发现 {before_missing_count} 个未返回结构化输出的评估项，开始针对性修复...")

        # 遍历进行修复
        for news in tqdm(news_list, desc="修复未返回结构化的评估"):
            evaluation = news.get("evaluation", {})
            if not evaluation:
                continue

            original_text = news.get("input_text", "")
            fused_text = news.get("fused_content", "")
            if not original_text.strip() or not fused_text.strip():
                continue

            global_eval = evaluation.get("global_evaluation", {})
            global_assessments = global_eval.get("global_assessments", [])
            needs_update = False

            for i, assess in enumerate(global_assessments):
                if self._is_missing_structure(assess):
                    problem_type = assess.get("problem_type")
                    if problem_type not in GLOBAL_PROBLEMS:
                        problem_type = "Logical Mainline Confusion"
                    logger.info(f"🔄 重新评估逻辑问题: {problem_type}")
                    new_result = self.evaluate_global_problem(
                        problem_type, original_text, fused_text
                    )
                    # 更新 assessment
                    global_assessments[i] = {
                        "problem_type": new_result["problem_type"],
                        "problem_exists": new_result.get("problem_exists", False),
                        "evidence": new_result.get("evidence", {}),
                        "evaluation_method": new_result.get("evaluation_method", ""),
                    }
                    needs_update = True

            # 如果这篇新闻被修复过，重新汇总统计信息
            if needs_update:
                # 重建 global_errors
                global_errors = []
                for assess in global_assessments:
                    if assess.get("problem_exists", False):
                        global_errors.append({
                            "error_type": assess["problem_type"],
                            "evidence": assess.get("evidence", {}),
                            "original_text": original_text,
                            "fused_text": fused_text,
                        })
                # 重新计算 total_global_problems
                total_global_problems = len(global_errors)
                
                global_eval["global_assessments"] = global_assessments
                global_eval["global_errors"] = global_errors
                global_eval["total_global_problems"] = total_global_problems
                
                # 重新生成 problem_classification 下的汇总数据
                global_classification = self._classify_global_problems(global_errors)
                problem_category_summary = self._summarize_problem_categories(global_assessments)
                factual_omissions_summary = self._summarize_factual_omissions(global_assessments)
                
                # 更新 overall evaluation 结果
                if "evaluation" in news:
                    news["evaluation"]["global_evaluation"] = global_eval
                    news["evaluation"]["problem_classification"] = {
                        "global_problems": global_classification,
                        "logical_problem_category_summary": problem_category_summary,
                        "factual_omissions_summary": factual_omissions_summary,
                        "total_problems": total_global_problems
                    }
                # 更新新闻顶层字段
                news["global_problem_count"] = total_global_problems
                news["total_problem_count"] = total_global_problems

        # 统计修复后的未结构化数量
        after_missing_count = 0
        for news in news_list:
            evaluation = news.get("evaluation", {})
            for assess in evaluation.get("global_evaluation", {}).get("global_assessments", []):
                if self._is_missing_structure(assess):
                    after_missing_count += 1

        return news_list, before_missing_count, after_missing_count

    def get_model_usage_stats(self):
        return self.model_stats.get_stats()

    def print_model_usage_summary(self):
        self.model_stats.print_summary()

# ==================== 处理器与工具函数 ====================

def create_evaluator(strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
    if strict_model_name is None:
        strict_model_name = DEFAULT_TEACHER_MODEL
    if cot_model_name is None:
        cot_model_name = strict_model_name
    strict_model_config = get_model_deployment_config(strict_model_name)
    cot_model_config = get_model_deployment_config(cot_model_name)
    return EvaluationSystem(strict_model_config, cot_model_config, gpu_device=gpu_device)

class EvaluationProcessor:
    def __init__(self, strict_model_name: str = None, cot_model_name: str = None, gpu_device: str = "cuda:0"):
        self.evaluator = create_evaluator(strict_model_name, cot_model_name, gpu_device=gpu_device)
        self.max_workers = max(1, MAX_WORKERS // 2)
        self.batch_size = max(1, BATCH_SIZE // 2)
        self.strict_model_name = strict_model_name or DEFAULT_TEACHER_MODEL
        self.cot_model_name = cot_model_name or self.strict_model_name

    def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            evaluation_result = self.evaluator.evaluate(news_item)
            return self._create_evaluated_news(news_item, evaluation_result)
        except Exception as e:
            logger.error(f"❌ 评估失败: {e}")
            return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

    def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluated_news = original_news.copy()
        evaluated_news.update({
            "evaluation": evaluation_result,
            "evaluation_success": evaluation_result.get("evaluation_success", False),
        })
        
        if evaluation_result.get("evaluation_success", False):
            evaluated_news.update({
                "global_problem_count": evaluation_result["global_evaluation"]["total_global_problems"],
                "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
            })
        
        return evaluated_news

    def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始批量评估 {len(news_list)} 篇新闻")
        processed_news = []
        success_count = 0
        total_global_problems = 0
        batch_size = min(self.batch_size, 2)
        
        for batch_start in range(0, len(news_list), batch_size):
            batch_end = min(batch_start + batch_size, len(news_list))
            batch = news_list[batch_start:batch_end]
            logger.info(f"🔄 评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
            
            batch_results = []
            pbar = tqdm(total=len(batch), desc=f"评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                future_to_news = {executor.submit(self.process_single_news, news): news for news in batch}
                
                for future in as_completed(future_to_news):
                    news_item = future_to_news[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result.get("evaluation_success", False):
                            success_count += 1
                            total_global_problems += result.get("global_problem_count", 0)
                            
                    except Exception as e:
                        logger.error(f"❌ 评估批次处理失败: {e}")
                        error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
                        batch_results.append(error_news)
                    finally:
                        if pbar:
                            pbar.update(1)
            
            if pbar:
                pbar.close()
            
            processed_news.extend(batch_results)
            batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
            logger.info(f"✅ 评估批次完成: 成功 {batch_success}/{len(batch_results)}")
        
        self.evaluator.print_model_usage_summary()
        logger.info(f"🎉 批量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
        logger.info(f"📊 总逻辑主线问题: {total_global_problems}")
        
        return processed_news

# ==================== 其他工具函数 ====================

def load_news_for_evaluation(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            logger.info(f"加载 {len(data)} 条新闻（旧格式）")
            return data
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                logger.info(f"加载到嵌套结构，顶层data包含 {len(data['data'])} 条记录")
                if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
                    first_item = data["data"][0]
                    if any(key in first_item for key in ["fused_content", "input_text", "alignment"]):
                        logger.info(f"直接使用顶层data中的 {len(data['data'])} 条新闻项")
                        return data["data"]
                    elif "data" in first_item and isinstance(first_item["data"], list):
                        inner_items = []
                        for outer_item in data["data"]:
                            if isinstance(outer_item, dict) and "data" in outer_item and isinstance(outer_item["data"], list):
                                inner_items.extend(outer_item["data"])
                        if inner_items:
                            logger.info(f"从嵌套结构中提取 {len(inner_items)} 条内层新闻项")
                            return inner_items
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
    new_base_name = f"{file_name_without_ext}_logic_{model_suffix}{file_ext}"
    return os.path.join(dir_name, new_base_name)

def save_evaluated_news(news_list: List[Dict[str, Any]], file_path: str, stats: Optional[Dict[str, Any]] = None, strict_model_name: str = "unknown", cot_model_name: str = "unknown"):
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        output_data = {
            "evaluation_stats": stats if stats else {},
            "strict_model": strict_model_name,
            "cot_model": cot_model_name,
            "total_items": len(news_list),
            "data": news_list
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 成功保存 {len(news_list)} 篇评估新闻到 {file_path}")
    except Exception as e:
        logger.error(f"❌ 保存文件失败: {e}")

def generate_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
    if not successful_evaluations:
        return {
            "summary": {
                "total_news_evaluated": 0,
                "average_global_problems": 0.0,
                "average_total_problems": 0.0,
            },
            "global_problems_classification": {"by_type": {}, "total_count": 0},
            "logical_problem_category_summary": {},
            "factual_omissions_summary": {}
        }
    
    total_news = len(successful_evaluations)
    global_problems_by_type = defaultdict(int)
    category_totals = defaultdict(int)
    factual_omissions_total = 0
    
    for news in successful_evaluations:
        evaluation = news.get("evaluation", {})
        problem_classification = evaluation.get("problem_classification", {})
        
        # 统计问题类型
        global_problems = problem_classification.get("global_problems", {})
        for problem_type, count in global_problems.get("by_type", {}).items():
            global_problems_by_type[problem_type] += count
        
        # 统计逻辑问题分类
        logical_summary = problem_classification.get("logical_problem_category_summary", {})
        category_counts = logical_summary.get("category_counts", {})
        for category, count in category_counts.items():
            category_totals[category] += count
        
        # 统计事实遗漏
        factual_summary = problem_classification.get("factual_omissions_summary", {})
        factual_omissions_total += factual_summary.get("factual_omissions_count", 0)
    
    avg_global_problems = sum(news.get("global_problem_count", 0) for news in successful_evaluations) / total_news
    avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
    
    # 转换为中文显示
    category_display = {}
    for eng_category, chn_name in LOGICAL_PROBLEM_CATEGORIES.items():
        if eng_category in category_totals:
            category_display[chn_name] = category_totals[eng_category]
    
    return {
        "summary": {
            "total_news_evaluated": total_news,
            "average_global_problems": round(avg_global_problems, 2),
            "average_total_problems": round(avg_total_problems, 2),
        },
        "global_problems_classification": {
            "by_type": dict(global_problems_by_type),
            "total_count": sum(global_problems_by_type.values())
        },
        "logical_problem_category_summary": {
            "category_totals": dict(category_totals),
            "category_display": category_display,
            "total_categories": len(category_totals)
        },
        "factual_omissions_summary": {
            "total_factual_omissions": factual_omissions_total,
            "average_factual_omissions": factual_omissions_total / total_news if total_news > 0 else 0
        }
    }

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="新闻逻辑主线评估工具")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
    parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
    parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
    parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子，默认 {DEFAULT_SEED}")
    parser.add_argument("--fix-mode", action="store_true", help="修复模式：识别并重新评估未返回结构化输出的项（在已有的输入评估文件基础上修复）")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    # 设置用户指定的随机种子
    if args.seed != DEFAULT_SEED:
        print(f"🔧 使用用户指定的随机种子: {args.seed}")
        set_global_seed(args.seed)

    print("📥 加载新闻数据...")
    news_data = load_news_for_evaluation(args.input)
    if not news_data:
        print("❌ 没有加载到新闻数据")
        sys.exit(1)

    if args.sample > 0:
        news_data = news_data[:args.sample]
        print(f"🔬 采样处理前 {args.sample} 篇新闻")

    # ====== 修复模式分支 ======
    if args.fix_mode:
        print("🛠️ 启动结构化输出修复模式...")
        # 创建 evaluator 实例
        strict_model_name = args.strict_model or DEFAULT_TEACHER_MODEL
        cot_model_name = args.cot_model or strict_model_name
        strict_model_config = get_model_deployment_config(strict_model_name)
        cot_model_config = get_model_deployment_config(cot_model_name)
        evaluator = EvaluationSystem(strict_model_config, cot_model_config, gpu_device=args.gpu_device)
        
        fixed_news_list, before_count, after_count = evaluator.fix_missing_evaluations(news_data)
        
        print("\n" + "="*50)
        print("✅ 修复阶段执行完成！")
        print(f"📊 修复前 - 识别到未返回结构化输出的数量: {before_count}")
        print(f"📊 修复后 - 遗留未返回结构化输出的数量: {after_count}")
        print(f"📈 本次成功补充修正数量: {before_count - after_count}")
        print("="*50 + "\n")
        
        # 保存修复后的更新数据
        stats = evaluator.get_model_usage_stats()
        output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)
        save_evaluated_news(fixed_news_list, output_path_with_suffix, stats, strict_model_name, cot_model_name)
        sys.exit(0)

    # ====== 正常评估模式 ======
    print("🚀 初始化评估处理器...")
    print(f"🎯 使用GPU设备: {args.gpu_device}")
    processor = EvaluationProcessor(
        strict_model_name=args.strict_model,
        cot_model_name=args.cot_model,
        gpu_device=args.gpu_device
    )

    print("🔄 开始逻辑主线评估...")
    evaluated_news = processor.process_news_batch(news_data)

    stats = processor.evaluator.get_model_usage_stats()

    strict_model_name = processor.strict_model_name
    cot_model_name = processor.cot_model_name
    output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)

    print("💾 保存结果...")
    save_evaluated_news(evaluated_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)

    summary = generate_summary_report(evaluated_news)
    print("\n📊 评估摘要:")
    print(f" 评估新闻总数: {summary['summary']['total_news_evaluated']}")
    print(f" 平均逻辑主线问题数: {summary['summary']['average_global_problems']}")
    print(f" 平均总问题数: {summary['summary']['average_total_problems']}")

    if summary['global_problems_classification']['total_count'] > 0:
        print(f" 逻辑主线问题分类: {summary['global_problems_classification']['by_type']}")
    
    if summary['logical_problem_category_summary']['total_categories'] > 0:
        print("\n🔍 逻辑问题分类统计:")
        for chn_name, count in summary['logical_problem_category_summary']['category_display'].items():
            print(f"  {chn_name}: {count}")
    
    if summary['factual_omissions_summary']['total_factual_omissions'] > 0:
        print(f"\n📝 事实遗漏统计: {summary['factual_omissions_summary']['total_factual_omissions']} 处")
        print(f"  平均每篇新闻事实遗漏: {summary['factual_omissions_summary']['average_factual_omissions']:.2f}")
    
    # 输出随机种子信息
    print(f"\n🌱 评估使用的随机种子: {args.seed}")

if __name__ == "__main__":
    main()