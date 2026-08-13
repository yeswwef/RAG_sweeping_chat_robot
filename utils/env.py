# utils/env.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 优先加载项目根目录的 .env（不依赖当前工作目录，从哪启动都能找到）
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_dotenv_path)


class Settings:
    """所有配置集中管理，以后新增配置只改这里"""

    # 通义千问 API Key
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    # 高德地图 API Key（用于天气查询）
    amap_api_key: str = os.getenv("AMAP_API_KEY", "")
    # 日志级别
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # ChromaDB 持久化目录
    chroma_persist_dir: str = "rag/chroma_db"


# 创建全局单例，其他地方直接 import 这个实例
settings = Settings()
