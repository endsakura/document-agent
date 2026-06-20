"""LLM 配置 — OpenAI 兼容 API（openai-hub）。"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 启动时加载项目根目录 .env（与 config.py 同级）
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# 默认走 openai-hub 代理；可用环境变量覆盖
DEFAULT_BASE_URL = "https://api.openai-hub.com/v1"
DEFAULT_MODEL = "gpt-4o"


def get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")


def get_base_url() -> str:
    return (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_yolo_model_path() -> str:
    """YOLO 权重路径，默认 models/best.pt（该文件不提交 GitHub）。"""
    return os.getenv("YOLO_MODEL_PATH", "models/best.pt")


def mask_key(key: Optional[str]) -> str:
    if not key:
        return "(未设置)"
    if len(key) <= 10:
        return "***"
    return f"{key[:7]}...{key[-4:]}"
