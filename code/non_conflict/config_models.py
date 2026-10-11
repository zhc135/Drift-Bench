import os

# config_models.py
from typing import Dict, Any

# 本地模型配置 (vLLM部署)
LOCAL_MODELS = {
    'gemma-3-12b-it': {
        "type": "local",
        "model": "gemma-3-12b-it",
        "base_url": "http://localhost:8007/v1",  # vLLM服务器地址
        "api_key": "EMPTY"  # vLLM通常不需要API密钥
    },
    "qwen-7b": {
        "type": "local",
        "model": "qwen-7b",
        "base_url": "http://localhost:8006/v1",  # vLLM服务器地址
        "api_key": "EMPTY"  # vLLM通常不需要API密钥
    },
    "gpt-oss-20b": {
        "type": "local", 
        "model": "gpt-oss-20b",
        "base_url": "http://localhost:8005/v1",
        "api_key": "EMPTY"
    },
    "qwen-05b": {
        "type": "local",
        "model": "qwen-05b",
        "base_url": "http://localhost:8002/v1",
        "api_key": "EMPTY"
    },
    "qwen-8b": {
        "type": "local",
        "model": "qwen-8b",
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY"
    },
    "llama3.1-8b-Instruct": {
        "type": "local",
        "model": "llama3.1-8b-Instruct",
        "base_url": "http://localhost:8005/v1",
        "api_key": "EMPTY"
    },
    "gemma-4bt": {
        "type": "local",
        "model": "gemma-4b",
        "base_url": "http://localhost:8004/v1",
        "api_key": "EMPTY"
    },
    "deepseek-R1-distill": {
        "type": "local",
        "model": "deepseek-R1-distill",
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY"
    },
    "phi-4": { 
        "type": "local",
        "model": "phi-4",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY"
    }

}

# 第三方API模型配置
THIRD_PARTY_MODELS = {
    "deepseek-v3.1": {
        "type": "third_party",
        "model": "deepseek-v3.1"
    },
    "gpt-5.4": {
        "type": "third_party",
        "model": "gpt-5.4"
    },
    "gemini-2.5-flash": {
        "type": "third_party",
        "model": "gemini-2.5-flash"
    },
    "gemini-2.5-pro": {
        "type": "third_party",
        "model": "gemini-2.5-pro"
    },
    "claude-sonnet-4-5-20250929": {
        "type": "third_party",
        "model": "claude-sonnet-4-5-20250929"
    },
    "doubao-seed-1-8-251228": {
        "type": "third_party",
        "model": "doubao-seed-1-8-251228"
    },
    "doubao-seed-1-6-250615": {
        "type": "third_party",
        "model": "doubao-seed-1-6-250615"
    },  
    "gpt-oss-120b": {
        "type": "third_party",
        "model": "gpt-oss-120b"
    },
    "gpt-4": {
        "type": "third_party",
        "model": "gpt-4"
    },
    "gpt-4o-mini": {
        "type": "third_party",
        "model": "gpt-4o-mini"
    },
    "gpt-4o": {
        "type": "third_party",
        "model": "gpt-4o"
    },
    "deepseek-chat": {
        "type": "third_party", 
        "model": "deepseek-chat"
    },
    "qwen-plus": {
        "type": "third_party",
        "model": "qwen-plus"
    },
    "gpt-3.5-turbo": {
        "type": "third_party",
        "model": "gpt-3.5-turbo"
    },
    "grok-2": {
        "type": "third_party",
        "model": "grok-2"
    },
    "gemini-2.5-flash": {
        "type": "third_party",
        "model": "gemini-2.5-flash"
    },
    'doubao-1-5-pro-256k-250115': {
        "type": "third_party",
        "model": "doubao-1-5-pro-256k-250115"
    },
    'doubao-1-5-pro-32k-250115': {
        "type": "third_party",
        "model": "doubao-1-5-pro-32k-250115"
    }
}

SIMILARITY_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "local_path": os.getenv(
            "DRIFT_SIMILARITY_MODEL_PATH",
            os.path.join(
                os.getenv("DRIFT_MODELS_ROOT", os.path.join("models")),
                "sentence-transformers",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ),
        )
    }
}

# 教师模型配置
TEACHER_MODELS ={
    "gpt-5.4": THIRD_PARTY_MODELS["gpt-5.4"],
    "gpt-4o-mini": THIRD_PARTY_MODELS["gpt-4o-mini"],
    "gpt-4o": THIRD_PARTY_MODELS["gpt-4o"],
    "deepseek-chat": THIRD_PARTY_MODELS["deepseek-chat"],
    "qwen-plus": THIRD_PARTY_MODELS["qwen-plus"],
    # "qwen-4b": LOCAL_MODELS["qwen-4b"], 
    "gpt-4":THIRD_PARTY_MODELS["gpt-4"],
    "gpt-oss-120b":THIRD_PARTY_MODELS["gpt-oss-120b"],
    "doubao-seed-1-8-251228":THIRD_PARTY_MODELS["doubao-seed-1-8-251228"],
    "doubao-seed-1-6-250615":THIRD_PARTY_MODELS["doubao-seed-1-6-250615"],
    "deepseek-v3.1":THIRD_PARTY_MODELS["deepseek-v3.1"],
    "llama3.1-8b-Instruct": LOCAL_MODELS["llama3.1-8b-Instruct"], 
    'doubao-1-5-pro-256k-250115': THIRD_PARTY_MODELS['doubao-1-5-pro-256k-250115'] # 本地部署的教师模型

}

# 学生模型配置
STUDENT_MODELS = {
    "gpt-5.4": THIRD_PARTY_MODELS["gpt-5.4"],
    'gemma-3-12b-it': LOCAL_MODELS['gemma-3-12b-it'],  # 本地部署的学生模型
    "gpt-3.5-turbo": THIRD_PARTY_MODELS["gpt-3.5-turbo"],
    "grok-2": THIRD_PARTY_MODELS["grok-2"],
    "gemini-2.5-flash": THIRD_PARTY_MODELS["gemini-2.5-flash"],
    "gemini-2.5-pro": THIRD_PARTY_MODELS["gemini-2.5-pro"],
    "claude-sonnet-4-5-20250929": THIRD_PARTY_MODELS["claude-sonnet-4-5-20250929"],
    "qwen-7b": LOCAL_MODELS["qwen-7b"],  # 本地部署的学生模型
    "qwen-8b": LOCAL_MODELS["qwen-8b"], 
    "qwen-05b": LOCAL_MODELS["qwen-05b"], 
    # "qwen-4b": LOCAL_MODELS["qwen-4b"], 
    "llama3.1-8b-Instruct": LOCAL_MODELS["llama3.1-8b-Instruct"],  # 本地部署的学生模型
    "deepseek-R1-distill":LOCAL_MODELS["deepseek-R1-distill"],
    "gemma-4bt":LOCAL_MODELS["gemma-4bt"],
    'phi-4': LOCAL_MODELS['phi-4'],  # 本地部署的学生模型
    'gpt-oss-20b': LOCAL_MODELS['gpt-oss-20b']  # 本地部署的学生模型
}

# 默认选择
DEFAULT_TEACHER_MODEL = "gpt-oss-120b"  # 修改为本地模型
DEFAULT_STUDENT_MODEL = "qwen-05b"  # 修改为本地模型
DEFAULT_SIMILARITY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 模型特定配置
MODEL_CONFIGS = {
    "deepseek-v3.1": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "gpt-5.4": {
        "temperature": 0.7,
        "max_tokens": 8000
    },
    "doubao-1-5-pro-256k-250115": {
        "temperature": 0.7,
        "max_tokens": 8000
    },
    "gpt-oss-120b": {
        "temperature": 0.7,
        "max_tokens": 8000
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "gpt-4": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "claude-sonnet-4-5-20250929": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "qwen-plus": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "qwen-7b": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "qwen-05b": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "deepseek-chat": {
        "temperature": 0.7,
        "max_tokens": 2500
    },
    "llama3.1-8b-Instruct": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "llama3-70b": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "grok-2": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "gemini-2.5-flash": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "gemini-2.5-pro": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "gpt3-text": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    'gemma-3-12b-it': {
        "temperature": 0.7,    
        "max_tokens": 3000
    } ,
    'gpt-oss-20b': {
        "temperature": 0.7,
        "max_tokens": 3000
    }       ,
    'doubao-seed-1-8-251228': {
        "temperature": 0.7,
        "max_tokens": 8000
    }   
}


# Example local vLLM launch template:
# CUDA_VISIBLE_DEVICES=<gpu_id> \
# python -m vllm.entrypoints.openai.api_server \
#     --model ${DRIFT_MODELS_ROOT:-models}/<model_path> \
#     --served-model-name <served_name> \
#     --trust-remote-code \
#     --dtype half \
#     --enforce-eager \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 32768 \
#     --port <port>


