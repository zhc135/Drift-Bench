"""Third-party API configuration for DRIFT-Bench non-conflict experiments.

The original experiment directory used private API keys. They are intentionally
not copied into this release. Set one or more environment variables below before
running scripts that call third-party models.
"""

import os
from itertools import cycle


def _api_config(key_env: str, base_url_env: str, default_base_url: str) -> dict:
    return {
        "api_key": os.getenv(key_env, ""),
        "base_url": os.getenv(base_url_env, default_base_url),
    }


THIRD_PARTY_CONFIGS = [
    _api_config("DRIFT_API_KEY_1", "DRIFT_API_BASE_1", "https://api.openai.com/v1"),
    _api_config("DRIFT_API_KEY_2", "DRIFT_API_BASE_2", "https://api.openai.com/v1"),
    _api_config("DRIFT_API_KEY_3", "DRIFT_API_BASE_3", "https://api.openai.com/v1"),
]

THIRD_PARTY_CONFIGS = [cfg for cfg in THIRD_PARTY_CONFIGS if cfg["api_key"]]

if not THIRD_PARTY_CONFIGS:
    THIRD_PARTY_CONFIGS = [
        {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
    ]

API_KEY_CYCLER = cycle(THIRD_PARTY_CONFIGS)


def get_api_config() -> dict:
    cfg = next(API_KEY_CYCLER)
    if not cfg.get("api_key"):
        raise RuntimeError(
            "No API key configured. Set DRIFT_API_KEY_1 or OPENAI_API_KEY before "
            "running third-party model evaluation/fusion scripts."
        )
    return cfg
