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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from sentence_transformers import SentenceTransformer
import hashlib

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

# 日志配置
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# 修复jieba缓存目录问题
jieba_cache_dir = os.path.join(os.path.expanduser("~"), ".jieba_cache")
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
if DEBUG_MODE:
    logger.info(f"🔧 设置jieba缓存目录: {jieba_cache_dir}")

# ==================== 辅助函数：新闻唯一标识 ====================

def get_news_hash(news: Dict[str, Any]) -> str:
    """基于新闻内容生成唯一哈希值，用于断点续传识别"""
    # 优先使用 input_text 和 fused_content，如果没有则使用整个新闻的 JSON 字符串
    input_text = news.get("input_text", "")
    fused_content = news.get("fused_content", "")
    # 如果新闻中已经有 _news_hash 字段，直接返回
    if "_news_hash" in news:
        return news["_news_hash"]
    # 否则基于内容计算
    content = f"{input_text}|{fused_content}"
    if not content.strip():
        # 保底：序列化整个新闻
        content = json.dumps(news, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

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
    "Lack OF Information": {
        "description": "融合灾害事件信息时，原始文本中的重要细节被遗漏，导致对事件的理解不完整。",
        "evaluation_method": "直接分析unused_content列表，判断遗漏每条信息是否影响融合文本与原始文本之间的意思一致性。如果unused_content中的任何信息对于保持原始意思很重要，则应被视为缺失信息。"
    }
}

PARAGRAPH_LEVEL_PROBLEMS = {
    # 可按需填充
    "Lack of Factual Completeness": {
        "description": "融合灾害事件信息时，原始文本中存在的关键事实细节（如受灾区域、伤亡人数、灾害等级或救援进展）缺失，导致对情况的理解不完整。",
        "evaluation_method": "比较段落级别的证据和融合文本，检查是否有缺失的关键事实。"
    }
}

# ==================== Prompt 模板 ====================

GLOBAL_EVALUATE_PROMPT = """
作为一名专业的新闻编辑，请评估以下融合文本是否包含'{problem_type}'问题。

## 问题描述: {problem_description}

## 评估要求:
1. **专注于unused_content分析**: 直接分析unused_content列表中的每条信息。
2. **评估逻辑**: 对于unused_content中的每个项目，判断遗漏它是否影响融合文本与原始文本之间的整体意思一致性。
3. **考虑因素**:
   - 这些信息对于理解关键事实、事件或上下文是否至关重要？
   - 遗漏这些信息是否会导致误解或不完整的理解？
   - 这些信息是否是冗余的或不重要的？
4. **决策**: 如果unused_content中的任何信息应该被包含以保持意思一致性，那么问题就存在。
5. **重要**: 无论问题是否存在，你必须提供所有unused_content项目的详细结构化分析。

## 文本内容:
**Original Text**: {original_text}
**Fused Text**: {fused_text}
**Unused Content**: {unused_content}

## 输出要求:
- 使用与输入文本相同的语言回答。
- 无论问题是否存在，始终以JSON格式提供结构化证据。

## 输出格式: 请以JSON格式返回评估结果，包含以下字段:
- "problem_exists": Boolean, 表示问题是否存在
- "evidence": 结构化分析，包含以下格式:
  - "analysis_summary": 评估的简要总结（分析了什么以及总体结论）
  - "unused_content_analysis": 对象列表，每个对象包含:
    - "content": 具体的未使用内容文本
    - "importance_level": 重要性等级（High/Medium/Low）
    - "omission_impact": 遗漏此内容是否影响意思（Yes/No/Partial）
    - "reasoning": 为什么应该或不应该包含的详细解释
    - "suggestion": 是否包含的建议
  - "overall_assessment": 关于信息完整性的总体结论
  - "critical_missing_info": 应该被包含的内容列表（如果没有则为空列表）
  - "non_critical_info": 正确省略的内容列表（附原因）
  - "final_verdict": 关于是否存在信息缺失的最终裁决（Yes/No/Partial）

## 重要说明:
- 无论problem_exists的值如何，始终在evidence字段中提供完整的结构化分析。
- 目标是确定遗漏unused_content是否影响意思一致性。
- 为每个未使用内容项目提供清晰详细的推理。
- 你的分析应该足够彻底，让阅读者能够准确理解为什么每个未使用内容项目被包含或未被包含。

请开始评估:
"""

PARAGRAPH_EVALUATE_PROMPT = """
作为一名专业的新闻编辑，请评估以下融合段落是否包含'{problem_type}'问题。

## 问题描述: {problem_description}

## 评估要求:
1. 仔细比较融合段落与原始证据。
2. 判断'{problem_type}'问题是否存在。
3. **重要**: 无论问题是否存在，你必须提供完整的结构化分析。
4. 考虑以下因素:
   - 融合段落中是否缺少任何关键事实信息？
   - 融合段落是否准确代表了证据中的所有重要细节？
   - 任何遗漏是否会导致误解或不完整的理解？
   - 是否有任何冗余或不重要的细节被正确省略？

## 段落信息:
**Fused Paragraph**: {fused_paragraph}
**Original Evidence**: {evidence_texts}

## 对齐信息:
- Source Versions: {sources}
- Conflict Resolution: {conflict_resolved}

## 输出要求:
- 使用与输入文本相同的语言回答。
- 无论问题是否存在，始终以JSON格式提供结构化证据。

## 输出格式: 请以JSON格式返回评估结果，包含以下字段:
- "problem_exists": Boolean, 表示问题是否存在
- "evidence": 结构化分析，包含以下格式:
  - "analysis_summary": 评估的简要总结（分析了什么以及总体结论）
  - "evidence_comparison": 对象列表，每个对象代表与融合段落比较的证据:
    - "evidence_text": 具体的证据文本
    - "included_in_fused": 该证据是否包含在融合段落中（Yes/No/Partial）
    - "importance_level": 重要性等级（High/Medium/Low）
    - "omission_impact": 如果未包含，遗漏此内容是否影响意思（Yes/No/Partial）
    - "reasoning": 为什么应该或不应该包含的详细解释
    - "suggestion": 是否包含的建议
  - "key_factual_elements": 应该存在的关键事实元素列表
  - "missing_factual_elements": 融合段落中缺失的事实元素列表（如果没有则为空）
  - "correctly_omitted_elements": 正确省略的元素列表（附原因）
  - "overall_assessment": 关于事实完整性的总体结论
  - "final_verdict": 关于是否存在事实信息缺失的最终裁决（Yes/No/Partial）

## 重要说明:
- 无论problem_exists的值如何，始终在evidence字段中提供完整的结构化分析。
- 目标是确定融合段落与证据相比是否保持了事实完整性。
- 为每条证据提供清晰详细的推理。
- 你的分析应该足够彻底，让阅读者能够准确理解为什么每条证据被包含或未被包含。

请开始评估:
"""

# ==================== 文本相似度计算器 ====================

class TextSimilarityCalculator:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", local_model_path: str = None, gpu_device: str = "cuda:0"):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.gpu_device = gpu_device
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            import torch
            if self.gpu_device and torch.cuda.is_available():
                try:
                    gpu_id = int(self.gpu_device.split(":")[1])
                    if gpu_id < torch.cuda.device_count():
                        self.device = torch.device(self.gpu_device)
                    else:
                        logger.warning(f"⚠️ 指定的GPU设备 {self.gpu_device} 不可用，使用CPU")
                        self.device = torch.device("cpu")
                except Exception:
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.local_model_path and os.path.exists(self.local_model_path):
                self.sentence_model = SentenceTransformer(self.local_model_path, device=self.device)
                logger.info(f"✅ 从本地路径加载句子转换模型: {self.local_model_path}")
            else:
                self.sentence_model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✅ 句子转换模型 {self.model_name} 初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 句子转换模型初始化失败: {e}, 将使用TF-IDF")
            self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_chinese, min_df=1, max_df=0.8, ngram_range=(1, 2)
        )

    def _tokenize_chinese(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if self.sentence_model is not None:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            else:
                return self.calculate_tfidf_similarity(text1, text2)
        except Exception as e:
            logger.error(f"❌ 语义相似度计算失败: {e}")
            return self.calculate_tfidf_similarity(text1, text2)

    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"❌ TF-IDF相似度计算失败: {e}")
            return 0.0

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            words1 = set(self._tokenize_chinese(text1))
            words2 = set(self._tokenize_chinese(text2))
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"❌ Jaccard相似度计算失败: {e}")
            return 0.0

    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if text1 == text2:
                return 1.0
            len1, len2 = len(text1), len(text2)
            max_len = max(len1, len2)
            if max_len == 0:
                return 1.0
            distance = self._levenshtein_distance(text1, text2)
            return 1.0 - (distance / max_len)
        except Exception as e:
            logger.error(f"❌ 编辑距离相似度计算失败: {e}")
            return 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def calculate_comprehensive_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        return {
            "semantic_similarity": self.calculate_semantic_similarity(text1, text2),
            "tfidf_similarity": self.calculate_tfidf_similarity(text1, text2),
            "jaccard_similarity": self.calculate_jaccard_similarity(text1, text2),
            "levenshtein_similarity": self.calculate_levenshtein_similarity(text1, text2),
        }

    def calculate_fused_text_similarities(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fused_content = news_item.get("fused_content", "")
            original_content = news_item.get("input_text", "")
            if not fused_content.strip() or not original_content.strip():
                logger.warning("❌ 融合文本或原始文本为空")
                return self._create_empty_similarity_result()
            similarities = {
                "original": self.calculate_comprehensive_similarity(fused_content, original_content)
            }
            rewrite_versions = {}
            for i in range(1, 4):
                rewrite_key_v = f"rewritten_v{i}"
                rewrite_key_n = f"rewrite_{i}"
                if rewrite_key_v in news_item and news_item[rewrite_key_v]:
                    rewrite_versions[rewrite_key_v] = news_item[rewrite_key_v]
                elif rewrite_key_n in news_item and news_item[rewrite_key_n]:
                    rewrite_versions[rewrite_key_n] = news_item[rewrite_key_n]
            for version_key, rewrite_content in rewrite_versions.items():
                if rewrite_content and rewrite_content.strip():
                    try:
                        similarities[version_key] = self.calculate_comprehensive_similarity(fused_content, rewrite_content)
                    except Exception as e:
                        logger.warning(f"⚠️ 计算与{version_key}的相似度失败: {e}")
                        similarities[version_key] = {"comprehensive_score": 0.0, "calculation_success": False}
            avg_similarities = self._calculate_average_similarities(similarities)
            analysis = self._generate_similarity_analysis(similarities, avg_similarities)
            return {
                "similarities": similarities,
                "average_similarities": avg_similarities,
                "analysis": analysis,
                "similarity_calculation_success": True,
            }
        except Exception as e:
            logger.error(f"❌ 相似度计算失败: {e}")
            return self._create_empty_similarity_result()

    def _calculate_average_similarities(self, similarities: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        avg_similarities = {}
        methods = ["semantic_similarity", "tfidf_similarity", "jaccard_similarity", "levenshtein_similarity"]
        for method in methods:
            scores = []
            for version_similarities in similarities.values():
                if method in version_similarities:
                    scores.append(version_similarities[method])
            if scores:
                avg_similarities[method] = sum(scores) / len(scores)
            else:
                avg_similarities[method] = 0.0
        return avg_similarities

    def _generate_similarity_analysis(self, similarities: Dict[str, Dict[str, float]], avg_similarities: Dict[str, float]) -> Dict[str, Any]:
        analysis = {"overall_assessment": "", "key_observations": [], "recommendations": []}
        semantic_avg = avg_similarities.get("semantic_similarity", 0)
        if semantic_avg >= 0.8:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本高度相似，信息保留完整"
        elif semantic_avg >= 0.6:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本中等相似，主要信息得到保留"
        elif semantic_avg >= 0.4:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本相似度较低，存在信息丢失"
        else:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本相似度很低，信息保留不完整"
        if "original" in similarities:
            orig_semantic = similarities["original"].get("semantic_similarity", 0)
            if orig_semantic < 0.5:
                analysis["key_observations"].append("融合文本与原始文本语义相似度较低，可能存在重要信息遗漏")
        method_scores = list(avg_similarities.values())
        if len(method_scores) >= 2:
            score_variance = np.var(method_scores)
            if score_variance > 0.1:
                analysis["key_observations"].append("不同相似度计算方法结果差异较大，建议关注语义相似度")
        if semantic_avg < 0.6:
            analysis["recommendations"].append("建议检查融合文本是否保留了原始文本的关键信息")
        if avg_similarities.get("jaccard_similarity", 0) < 0.3:
            analysis["recommendations"].append("词汇重叠度较低，建议检查是否使用了过多不同的表达方式")
        return analysis

    def _create_empty_similarity_result(self) -> Dict[str, Any]:
        return {
            "similarities": {},
            "average_similarities": {},
            "analysis": {
                "overall_assessment": "相似度计算失败",
                "key_observations": ["无法计算相似度"],
                "recommendations": ["请检查输入文本"],
            },
            "similarity_calculation_success": False,
        }

# ==================== 双重评估系统 ====================

class DualEvaluationSystem:
    class SimpleResponse:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
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
            logger.info(f"🤖 初始化双重评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
            logger.info(f"🎯 指定GPU设备: {gpu_device}")

        self.llm_strict = self._initialize_llm(strict_model_config)
        self.llm_cot = self._initialize_llm(self.cot_model_config)
        self.similarity_calculator = TextSimilarityCalculator(local_model_path=local_similarity_model_path, gpu_device=gpu_device)

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 12288)
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
                return DualEvaluationSystem.SimpleResponse(text)
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

    # ==================== 整体评估 ====================

    def evaluate_global_problem(self, problem_type: str, original_text: str, fused_text: str, unused_content: str) -> Dict[str, Any]:
        try:
            problem_info = GLOBAL_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            prompt = GLOBAL_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                original_text=original_text,
                fused_text=fused_text,
                unused_content=unused_content,
            )
            messages = [
                SystemMessage(content="你是一名专业的新闻编辑，负责评估文本融合的整体质量。特别注意：对于'信息缺失'问题，直接分析unused_content中的内容，判断其缺失是否影响文本意思的一致性。无论问题是否存在，都必须提供完整的结构化分析证据。"),
                HumanMessage(content=prompt),
            ]
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"global_{problem_type}")
            if not response:
                return {
                    "problem_exists": False,
                    "problem_type": problem_type,
                    "evidence": {
                        "analysis_summary": "评估失败，无法获取模型响应",
                        "unused_content_analysis": [],
                        "overall_assessment": "评估失败",
                        "critical_missing_info": [],
                        "non_critical_info": [],
                        "final_verdict": "无法确定"
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
                    "unused_content_analysis": [],
                    "overall_assessment": "证据缺失",
                    "critical_missing_info": [],
                    "non_critical_info": [],
                    "final_verdict": "无法确定"
                }
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
                    "unused_content_analysis": [],
                    "overall_assessment": "评估失败",
                    "critical_missing_info": [],
                    "non_critical_info": [],
                    "final_verdict": "无法确定"
                },
                "evaluation_method": "global_assessment"
            }

    # ==================== 段落级别评估 ====================

    def evaluate_paragraph_problem(self, problem_type: str, paragraph_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            problem_info = PARAGRAPH_LEVEL_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            fused_paragraph = paragraph_data.get("text", "")
            evidence_texts = paragraph_data.get("evidence", [])
            sources = paragraph_data.get("sources", [])
            conflict_resolved = paragraph_data.get("conflict_resolved", "未处理")
            prompt = PARAGRAPH_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                fused_paragraph=fused_paragraph,
                evidence_texts=json.dumps(evidence_texts, ensure_ascii=False, indent=2),
                sources=sources,
                conflict_resolved=conflict_resolved,
            )
            messages = [
                SystemMessage(content="你是一名专业的新闻编辑，负责评估段落级别的融合质量，无论问题是否存在，都必须提供完整的结构化分析证据。"),
                HumanMessage(content=prompt),
            ]
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict, step_name=f"paragraph_{problem_type}")
            if not response:
                return {
                    "problem_exists": False,
                    "problem_type": problem_type,
                    "evidence": {
                        "analysis_summary": "评估失败，无法获取模型响应",
                        "evidence_comparison": [],
                        "key_factual_elements": [],
                        "missing_factual_elements": [],
                        "correctly_omitted_elements": [],
                        "overall_assessment": "评估失败",
                        "final_verdict": "无法确定"
                    },
                    "paragraph_idx": paragraph_data.get("sent_idx"),
                    "evaluation_method": "paragraph_assessment"
                }
            eval_data = self.safe_json_parse(response.content)
            problem_exists = eval_data.get("problem_exists", False)
            evidence = eval_data.get("evidence", {})
            if not evidence:
                evidence = {
                    "analysis_summary": "模型未返回结构化证据",
                    "evidence_comparison": [],
                    "key_factual_elements": [],
                    "missing_factual_elements": [],
                    "correctly_omitted_elements": [],
                    "overall_assessment": "证据缺失",
                    "final_verdict": "无法确定"
                }
            return {
                "problem_exists": bool(problem_exists),
                "problem_type": problem_type,
                "evidence": evidence,
                "paragraph_idx": paragraph_data.get("sent_idx"),
                "evaluation_method": "paragraph_assessment"
            }
        except Exception as e:
            logger.error(f"❌ 段落评估失败 {problem_type}: {e}")
            return {
                "problem_exists": False,
                "problem_type": problem_type,
                "evidence": {
                    "analysis_summary": f"评估异常: {str(e)}",
                    "evidence_comparison": [],
                    "key_factual_elements": [],
                    "missing_factual_elements": [],
                    "correctly_omitted_elements": [],
                    "overall_assessment": "评估失败",
                    "final_verdict": "无法确定"
                },
                "paragraph_idx": paragraph_data.get("sent_idx"),
                "evaluation_method": "paragraph_assessment"
            }

    def paragraph_evaluation_stage(self, alignment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("📝 阶段2: 段落级别评估")
        if not alignment_data:
            return {"paragraph_errors": [], "total_paragraph_problems": 0}
        filtered_alignment_data = [item for item in alignment_data if "unused_content" not in item]
        if not filtered_alignment_data:
            return {"paragraph_errors": [], "total_paragraph_problems": 0}
        if not PARAGRAPH_LEVEL_PROBLEMS:
            logger.info("没有启用的段落级别问题类型，跳过段落级别评估")
            return {"paragraph_errors": [], "total_paragraph_problems": 0}
        paragraph_errors = []
        paragraph_assessments = []
        evaluation_tasks = []
        for paragraph in filtered_alignment_data:
            for problem_type in PARAGRAPH_LEVEL_PROBLEMS.keys():
                evaluation_tasks.append({
                    "problem_type": problem_type,
                    "paragraph_data": paragraph
                })
        if not evaluation_tasks:
            return {"paragraph_errors": [], "total_paragraph_problems": 0}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(evaluation_tasks))) as executor:
            future_to_task = {
                executor.submit(self.evaluate_paragraph_problem, task["problem_type"], task["paragraph_data"]): task
                for task in evaluation_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    paragraph_assessments.append({
                        "problem_type": result["problem_type"],
                        "problem_exists": result.get("problem_exists", False),
                        "evidence": result.get("evidence", {}),
                        "paragraph_idx": result.get("paragraph_idx"),
                        "evaluation_method": result.get("evaluation_method", ""),
                    })
                    if result.get("problem_exists", False):
                        paragraph_errors.append({
                            "error_type": result["problem_type"],
                            "evidence": result.get("evidence", {}),
                            "paragraph_idx": result.get("paragraph_idx"),
                            "paragraph_text": task["paragraph_data"].get("text", ""),
                            "evaluation_method": result.get("evaluation_method", "")
                        })
                except Exception as e:
                    logger.error(f"❌ 段落级别评估异常: {e}")
        return {
            "paragraph_assessments": paragraph_assessments,
            "paragraph_errors": paragraph_errors,
            "total_paragraph_problems": len(paragraph_errors)
        }

    # ==================== 相似度分析 ====================

    def similarity_analysis_stage(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔍 阶段3: 相似度分析")
        try:
            similarity_result = self.similarity_calculator.calculate_fused_text_similarities(news_item)
            return similarity_result
        except Exception as e:
            logger.error(f"❌ 相似度分析失败: {e}")
            return self.similarity_calculator._create_empty_similarity_result()

    # ==================== 对齐数据提取 ====================

    def extract_alignment_data(self, fused_content: str) -> Tuple[List[Dict[str, Any]], str]:
        try:
            alignment_separator = "====ALIGNMENT===="
            if alignment_separator in fused_content:
                parts = fused_content.split(alignment_separator)
                if len(parts) > 1:
                    alignment_text = parts[1].strip()
                    alignment_data = json.loads(alignment_text)
                    unused_content = ""
                    for item in alignment_data:
                        if isinstance(item, dict) and "unused_content" in item:
                            unused_content = json.dumps(item["unused_content"], ensure_ascii=False, indent=2)
                            alignment_data = [i for i in alignment_data if "unused_content" not in i]
                            break
                    return alignment_data, unused_content
            return [], ""
        except Exception as e:
            logger.error(f"❌ 提取对齐数据失败: {e}")
            return [], ""

    # ==================== 主评估流程 ====================

    def dual_evaluation(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔄 开始双重评估（整体文本 + 段落级别）")
        original_text = news_item.get("input_text", "")
        fused_text = news_item.get("fused_content", "")
        if not original_text.strip() or not fused_text.strip():
            logger.warning(f"❌ 新闻内容为空，无法评估")
            return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
        alignment_data, unused_content = self.extract_alignment_data(fused_text)
        global_results = self.global_evaluation_stage(original_text, fused_text, unused_content)
        paragraph_results = self.paragraph_evaluation_stage(alignment_data)
        similarity_results = self.similarity_analysis_stage(news_item)
        global_classification = self._classify_global_problems(global_results["global_errors"])
        paragraph_classification = self._classify_paragraph_problems(paragraph_results["paragraph_errors"])
        result = {
            "evaluation_success": True,
            "global_evaluation": global_results,
            "paragraph_evaluation": paragraph_results,
            "similarity_analysis": similarity_results,
            "problem_classification": {
                "global_problems": global_classification,
                "paragraph_problems": paragraph_classification,
                "total_problems": global_results["total_global_problems"] + paragraph_results["total_paragraph_problems"]
            },
            "model_usage_stats": self.model_stats.get_stats()
        }
        logger.info("✅ 双重评估完成")
        if similarity_results.get("similarity_calculation_success", False):
            avg_similarities = similarity_results.get("average_similarities", {})
            semantic_sim = avg_similarities.get("semantic_similarity", 0)
            logger.info(f" 平均语义相似度: {semantic_sim:.3f}")
        return result

    def global_evaluation_stage(self, original_text: str, fused_text: str, unused_content: str = "") -> Dict[str, Any]:
        logger.info("🌐 阶段1: 整体文本评估")
        if not original_text.strip() or not fused_text.strip():
            return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        if not GLOBAL_PROBLEMS:
            logger.info("没有启用的整体问题类型，跳过整体评估")
            return {"global_assessments": [], "global_errors": [], "total_global_problems": 0}
        global_assessments = []
        global_errors = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(GLOBAL_PROBLEMS))) as executor:
            future_to_problem = {
                executor.submit(self.evaluate_global_problem, problem_type, original_text, fused_text, unused_content): problem_type
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
                            "unused_content_analysis": [],
                            "overall_assessment": "评估失败",
                            "critical_missing_info": [],
                            "non_critical_info": [],
                            "final_verdict": "无法确定"
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

    def _classify_paragraph_problems(self, paragraph_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        classification = defaultdict(int)
        for error in paragraph_errors:
            error_type = error.get("error_type", "未知类型")
            classification[error_type] += 1
        return {"by_type": dict(classification), "total_count": len(paragraph_errors)}

    def get_model_usage_stats(self):
        return self.model_stats.get_stats()

    def print_model_usage_summary(self):
        self.model_stats.print_summary()

# ==================== 处理器与工具函数 ====================

def create_dual_evaluator(strict_model_name: str = None, cot_model_name: str = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
    if strict_model_name is None:
        strict_model_name = DEFAULT_TEACHER_MODEL
    if cot_model_name is None:
        cot_model_name = strict_model_name
    strict_model_config = get_model_deployment_config(strict_model_name)
    cot_model_config = get_model_deployment_config(cot_model_name)
    return DualEvaluationSystem(strict_model_config, cot_model_config, local_similarity_model_path=local_similarity_model_path, gpu_device=gpu_device)

class DualEvaluationProcessor:
    def __init__(self, strict_model_name: str = None, cot_model_name: str = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
        self.evaluator = create_dual_evaluator(strict_model_name, cot_model_name, local_similarity_model_path=local_similarity_model_path, gpu_device=gpu_device)
        self.max_workers = max(1, MAX_WORKERS // 2)
        self.batch_size = max(1, BATCH_SIZE // 2)
        self.strict_model_name = strict_model_name or DEFAULT_TEACHER_MODEL
        self.cot_model_name = cot_model_name or self.strict_model_name

    def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            evaluation_result = self.evaluator.dual_evaluation(news_item)
            return self._create_evaluated_news(news_item, evaluation_result)
        except Exception as e:
            logger.error(f"❌ 双重评估失败: {e}")
            return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

    def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluated_news = original_news.copy()
        evaluated_news.update({
            "dual_evaluation": evaluation_result,
            "evaluation_success": evaluation_result.get("evaluation_success", False),
        })
        if evaluation_result.get("evaluation_success", False):
            evaluated_news.update({
                "global_problem_count": evaluation_result["global_evaluation"]["total_global_problems"],
                "paragraph_problem_count": evaluation_result["paragraph_evaluation"]["total_paragraph_problems"],
                "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
            })
        similarity_analysis = evaluation_result.get("similarity_analysis", {})
        if similarity_analysis.get("similarity_calculation_success", False):
            avg_similarities = similarity_analysis.get("average_similarities", {})
            evaluated_news.update({
                "semantic_similarity": avg_similarities.get("semantic_similarity", 0),
                "tfidf_similarity": avg_similarities.get("tfidf_similarity", 0),
                "jaccard_similarity": avg_similarities.get("jaccard_similarity", 0),
            })
        # 添加哈希值，便于断点续传识别
        evaluated_news["_news_hash"] = get_news_hash(original_news)
        return evaluated_news

    def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始批量双重评估 {len(news_list)} 篇新闻")
        processed_news = []
        success_count = 0
        total_global_problems = 0
        total_paragraph_problems = 0
        total_semantic_similarity = 0
        batch_size = min(self.batch_size, 2)
        for batch_start in range(0, len(news_list), batch_size):
            batch_end = min(batch_start + batch_size, len(news_list))
            batch = news_list[batch_start:batch_end]
            logger.info(f"🔄 双重评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
            batch_results = []
            pbar = tqdm(total=len(batch), desc=f"双重评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
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
                            total_paragraph_problems += result.get("paragraph_problem_count", 0)
                            semantic_sim = result.get("semantic_similarity", 0)
                            total_semantic_similarity += semantic_sim
                    except Exception as e:
                        logger.error(f"❌ 双重评估批次处理失败: {e}")
                        error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
                        batch_results.append(error_news)
                    finally:
                        if pbar:
                            pbar.update(1)
            if pbar:
                pbar.close()
            processed_news.extend(batch_results)
            batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
            logger.info(f"✅ 双重评估批次完成: 成功 {batch_success}/{len(batch_results)}")
        avg_semantic_similarity = total_semantic_similarity / success_count if success_count > 0 else 0
        self.evaluator.print_model_usage_summary()
        logger.info(f"🎉 双重批量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
        logger.info(f"📊 总整体问题: {total_global_problems}, 总段落问题: {total_paragraph_problems}")
        logger.info(f"🔍 平均语义相似度: {avg_semantic_similarity:.3f}")
        return processed_news

    # ---------------- 修复结构化输出失败的方法 ----------------
    def _is_missing_structure(self, assessment: Dict[str, Any]) -> bool:
        """判定某一项评估是否因未返回结构化输出而失败"""
        evidence = assessment.get("evidence", {})
        if not evidence:
            return True
        summary = evidence.get("analysis_summary", "")
        if not isinstance(summary, str):
            return True
        if "模型未返回结构化证据" in summary or "评估失败" in summary or "评估异常" in summary:
            return True
        return False

    def fix_missing_evaluations(self, news_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        """识别并仅对缺失结构化输出的部分发起重评请求"""
        before_missing_count = 0
        
        # 统计修复前的未结构化数量
        for news in news_list:
            dual_eval = news.get("dual_evaluation")
            if not dual_eval: continue
            for assess in dual_eval.get("global_evaluation", {}).get("global_assessments", []):
                if self._is_missing_structure(assess): before_missing_count += 1
            for assess in dual_eval.get("paragraph_evaluation", {}).get("paragraph_assessments", []):
                if self._is_missing_structure(assess): before_missing_count += 1

        if before_missing_count == 0:
            logger.info("🎉 没有发现未返回结构化输出的评估项，无需修复。")
            return news_list, 0, 0

        logger.info(f"🔧 发现 {before_missing_count} 个未返回结构化输出的评估项，开始针对性修复...")

        # 遍历进行修复
        for news in tqdm(news_list, desc="修复未返回结构化的评估"):
            dual_eval = news.get("dual_evaluation")
            if not dual_eval: continue

            original_text = news.get("input_text", "")
            fused_text = news.get("fused_content", "")
            alignment_data, unused_content = self.evaluator.extract_alignment_data(fused_text)
            filtered_alignment = [item for item in alignment_data if "unused_content" not in item]
            alignment_dict = {item.get("sent_idx"): item for item in filtered_alignment}

            needs_update = False

            # 1. 修复全局评估
            global_eval = dual_eval.get("global_evaluation", {})
            global_assessments = global_eval.get("global_assessments", [])
            for i, assess in enumerate(global_assessments):
                if self._is_missing_structure(assess):
                    problem_type = assess.get("problem_type")
                    logger.info(f"🔄 重新评估全局问题: {problem_type}")
                    new_assess = self.evaluator.evaluate_global_problem(
                        problem_type, original_text, fused_text, unused_content
                    )
                    global_assessments[i] = new_assess
                    needs_update = True

            # 2. 修复段落级评估
            para_eval = dual_eval.get("paragraph_evaluation", {})
            para_assessments = para_eval.get("paragraph_assessments", [])
            for i, assess in enumerate(para_assessments):
                if self._is_missing_structure(assess):
                    problem_type = assess.get("problem_type")
                    para_idx = assess.get("paragraph_idx")
                    para_data = alignment_dict.get(para_idx)
                    if para_data:
                        logger.info(f"🔄 重新评估段落问题: {problem_type} (Idx: {para_idx})")
                        new_assess = self.evaluator.evaluate_paragraph_problem(
                            problem_type, para_data
                        )
                        para_assessments[i] = new_assess
                        needs_update = True
                    else:
                        logger.warning(f"⚠️ 找不到对应的段落数据 Idx: {para_idx}")

            # 3. 如果这篇新闻被修复过，重新汇总 error 列表与问题类型统计
            if needs_update:
                # 重新构建全局 error 列表
                new_global_errors = []
                for assess in global_assessments:
                    if assess.get("problem_exists"):
                        new_global_errors.append({
                            "error_type": assess.get("problem_type"),
                            "evidence": assess.get("evidence", {}),
                            "original_text": original_text,
                            "fused_text": fused_text,
                        })
                global_eval["global_errors"] = new_global_errors
                global_eval["total_global_problems"] = len(new_global_errors)

                # 重新构建段落 error 列表
                new_para_errors = []
                for assess in para_assessments:
                    if assess.get("problem_exists"):
                        para_idx = assess.get("paragraph_idx")
                        para_text = alignment_dict.get(para_idx, {}).get("text", "")
                        new_para_errors.append({
                            "error_type": assess.get("problem_type"),
                            "evidence": assess.get("evidence", {}),
                            "paragraph_idx": para_idx,
                            "paragraph_text": para_text,
                            "evaluation_method": assess.get("evaluation_method", "")
                        })
                para_eval["paragraph_errors"] = new_para_errors
                para_eval["total_paragraph_problems"] = len(new_para_errors)

                # 重构最终统计
                dual_eval["problem_classification"] = {
                    "global_problems": self.evaluator._classify_global_problems(new_global_errors),
                    "paragraph_problems": self.evaluator._classify_paragraph_problems(new_para_errors),
                    "total_problems": len(new_global_errors) + len(new_para_errors)
                }
                news["global_problem_count"] = len(new_global_errors)
                news["paragraph_problem_count"] = len(new_para_errors)
                news["total_problem_count"] = len(new_global_errors) + len(new_para_errors)

        # 统计修复后的未结构化数量
        after_missing_count = 0
        for news in news_list:
            dual_eval = news.get("dual_evaluation")
            if not dual_eval: continue
            for assess in dual_eval.get("global_evaluation", {}).get("global_assessments", []):
                if self._is_missing_structure(assess): after_missing_count += 1
            for assess in dual_eval.get("paragraph_evaluation", {}).get("paragraph_assessments", []):
                if self._is_missing_structure(assess): after_missing_count += 1

        return news_list, before_missing_count, after_missing_count

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
                    if any(key in first_item for key in ["fused_content", "input_text", "alignment", "dual_evaluation"]):
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
    new_base_name = f"{file_name_without_ext}_{model_suffix}{file_ext}"
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

def generate_dual_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
    if not successful_evaluations:
        return {
            "summary": {
                "total_news_evaluated": 0,
                "average_global_problems": 0.0,
                "average_paragraph_problems": 0.0,
                "average_total_problems": 0.0,
                "average_semantic_similarity": 0.0
            },
            "global_problems_classification": {"by_type": {}, "total_count": 0},
            "paragraph_problems_classification": {"by_type": {}, "total_count": 0},
            "similarity_distribution": {
                "min_semantic_similarity": 0,
                "max_semantic_similarity": 0,
                "avg_semantic_similarity": 0,
            },
        }
    total_news = len(successful_evaluations)
    global_problems_by_type = defaultdict(int)
    paragraph_problems_by_type = defaultdict(int)
    semantic_similarities = []
    for news in successful_evaluations:
        evaluation = news.get("dual_evaluation", {})
        problem_classification = evaluation.get("problem_classification", {})
        global_problems = problem_classification.get("global_problems", {})
        for problem_type, count in global_problems.get("by_type", {}).items():
            global_problems_by_type[problem_type] += count
        paragraph_problems = problem_classification.get("paragraph_problems", {})
        for problem_type, count in paragraph_problems.get("by_type", {}).items():
            paragraph_problems_by_type[problem_type] += count
        semantic_sim = news.get("semantic_similarity", 0)
        semantic_similarities.append(semantic_sim)
    avg_global_problems = sum(news.get("global_problem_count", 0) for news in successful_evaluations) / total_news
    avg_paragraph_problems = sum(news.get("paragraph_problem_count", 0) for news in successful_evaluations) / total_news
    avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0
    return {
        "summary": {
            "total_news_evaluated": total_news,
            "average_global_problems": round(avg_global_problems, 2),
            "average_paragraph_problems": round(avg_paragraph_problems, 2),
            "average_total_problems": round(avg_total_problems, 2),
            "average_semantic_similarity": round(avg_semantic_similarity, 3),
        },
        "global_problems_classification": {
            "by_type": dict(global_problems_by_type),
            "total_count": sum(global_problems_by_type.values())
        },
        "paragraph_problems_classification": {
            "by_type": dict(paragraph_problems_by_type),
            "total_count": sum(paragraph_problems_by_type.values())
        },
        "similarity_distribution": {
            "min_semantic_similarity": min(semantic_similarities) if semantic_similarities else 0,
            "max_semantic_similarity": max(semantic_similarities) if semantic_similarities else 0,
            "avg_semantic_similarity": round(avg_semantic_similarity, 3),
        },
    }

# ==================== 主函数（带断点续传与修复模式） ====================

def main():
    parser = argparse.ArgumentParser(description="双重新闻评估工具（整体文本 + 段落级别评估）支持断点续传及特定错误修复")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径（若为修复模式，则为已生成的包含评估结果的文件）")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
    parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
    parser.add_argument("--similarity-model-path", "-smp", default=None, help="本地相似度模型路径")
    parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
    parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
    parser.add_argument("--batch-size", "-bs", type=int, default=2, help="断点续传时每批处理的新闻数量，默认2")
    parser.add_argument("--fix-mode", action="store_true", help="修复模式：识别并重新评估未返回结构化输出的项（在已有的输入评估文件基础上修复）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    print("📥 加载新闻数据...")
    news_data = load_news_for_evaluation(args.input)
    if not news_data:
        print("❌ 没有加载到新闻数据")
        sys.exit(1)

    if args.sample > 0:
        news_data = news_data[:args.sample]
        print(f"🔬 采样处理前 {args.sample} 篇新闻")

    strict_model_name = args.strict_model or DEFAULT_TEACHER_MODEL
    cot_model_name = args.cot_model or strict_model_name

    # ====== 修复模式分支 ======
    if args.fix_mode:
        print("🛠️ 启动结构化输出修复模式...")
        processor = DualEvaluationProcessor(
            strict_model_name=args.strict_model,
            cot_model_name=args.cot_model,
            local_similarity_model_path=args.similarity_model_path,
            gpu_device=args.gpu_device
        )
        
        fixed_news_list, before_count, after_count = processor.fix_missing_evaluations(news_data)
        
        print("\n" + "="*50)
        print("✅ 修复阶段执行完成！")
        print(f"📊 修复前 - 识别到未返回结构化输出的数量: {before_count}")
        print(f"📊 修复后 - 遗留未返回结构化输出的数量: {after_count}")
        print(f"📈 本次成功补充修正数量: {before_count - after_count}")
        print("="*50 + "\n")
        
        # 保存修复后的更新数据
        stats = processor.evaluator.get_model_usage_stats()
        output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)
        save_evaluated_news(fixed_news_list, output_path_with_suffix, stats, strict_model_name, cot_model_name)
        sys.exit(0)

    # ====== 常规处理与断点续传分支 ======
    output_path_with_suffix = add_model_suffix_to_filename(args.output, strict_model_name, cot_model_name)

    existing_news_by_hash = {}
    existing_news_list = []
    if os.path.exists(output_path_with_suffix):
        print(f"📂 发现已有输出文件 {output_path_with_suffix}，尝试加载已处理的结果...")
        try:
            with open(output_path_with_suffix, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if "data" in existing_data and isinstance(existing_data["data"], list):
                existing_news_list = existing_data["data"]
                for news in existing_news_list:
                    # 如果新闻没有哈希字段，则计算并补充（向后兼容）
                    if "_news_hash" not in news:
                        news["_news_hash"] = get_news_hash(news)
                    existing_news_by_hash[news["_news_hash"]] = news
                print(f"✅ 已加载 {len(existing_news_list)} 篇已评估新闻")
            else:
                print("⚠️ 输出文件格式不正确，将重新开始评估")
        except Exception as e:
            print(f"⚠️ 加载已有输出文件失败: {e}，将重新开始评估")

    # 确定待处理的新闻（基于哈希值去重）
    pending_news = []
    for news in news_data:
        h = get_news_hash(news)
        if h not in existing_news_by_hash:
            pending_news.append(news)
        else:
            logger.debug(f"新闻已处理，跳过: {h[:8]}...")

    print(f"📊 总新闻数: {len(news_data)}, 已处理: {len(existing_news_list)}, 待处理: {len(pending_news)}")
    if not pending_news:
        print("✅ 所有新闻均已处理，无需继续。")
        summary = generate_dual_summary_report(existing_news_list)
        print("\n📊 双重评估摘要:")
        print(f" 评估新闻总数: {summary['summary']['total_news_evaluated']}")
        print(f" 平均整体问题数: {summary['summary']['average_global_problems']}")
        print(f" 平均段落问题数: {summary['summary']['average_paragraph_problems']}")
        print(f" 平均总问题数: {summary['summary']['average_total_problems']}")
        print(f" 平均语义相似度: {summary['summary']['average_semantic_similarity']:.3f}")
        return

    # 初始化处理器
    print("🚀 初始化双重评估处理器...")
    print(f"🎯 使用GPU设备: {args.gpu_device}")
    processor = DualEvaluationProcessor(
        strict_model_name=args.strict_model,
        cot_model_name=args.cot_model,
        local_similarity_model_path=args.similarity_model_path,
        gpu_device=args.gpu_device
    )

    batch_size = max(1, args.batch_size)
    all_processed_news = existing_news_list.copy() 
    total_pending = len(pending_news)
    processed_in_this_run = 0

    for batch_start in range(0, total_pending, batch_size):
        batch_end = min(batch_start + batch_size, total_pending)
        batch = pending_news[batch_start:batch_end]
        print(f"\n🔄 处理批次 {batch_start//batch_size + 1}/{(total_pending + batch_size - 1)//batch_size}，新闻数 {len(batch)}")

        batch_results = processor.process_news_batch(batch)
        processed_in_this_run += len(batch_results)

        all_processed_news.extend(batch_results)

        stats = processor.evaluator.get_model_usage_stats()
        save_evaluated_news(all_processed_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)
        print(f"💾 已保存中间结果，累计完成 {len(all_processed_news)} 篇新闻")

    stats = processor.evaluator.get_model_usage_stats()
    save_evaluated_news(all_processed_news, output_path_with_suffix, stats, strict_model_name, cot_model_name)

    summary = generate_dual_summary_report(all_processed_news)
    print("\n📊 双重评估摘要:")
    print(f" 评估新闻总数: {summary['summary']['total_news_evaluated']}")
    print(f" 平均整体问题数: {summary['summary']['average_global_problems']}")
    print(f" 平均段落问题数: {summary['summary']['average_paragraph_problems']}")
    print(f" 平均总问题数: {summary['summary']['average_total_problems']}")
    print(f" 平均语义相似度: {summary['summary']['average_semantic_similarity']:.3f}")

    if summary['global_problems_classification']['total_count'] > 0:
        print(f" 整体问题分类: {summary['global_problems_classification']['by_type']}")

    if summary['paragraph_problems_classification']['total_count'] > 0:
        print(f" 段落问题分类: {summary['paragraph_problems_classification']['by_type']}")

    similarity_dist = summary['similarity_distribution']
    print(f" 语义相似度分布: 最低{similarity_dist['min_semantic_similarity']:.3f}, 最高{similarity_dist['max_semantic_similarity']:.3f}, 平均{similarity_dist['avg_semantic_similarity']:.3f}")

    processor.evaluator.print_model_usage_summary()

if __name__ == "__main__":
    main()