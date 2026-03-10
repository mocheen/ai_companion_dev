"""
工具函数
包含配置加载、日志配置等功能
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典，失败返回空字典
    """
    # 优先使用本地配置文件
    local_config_path = config_path.replace(".yaml", "_local.yaml")
    if os.path.exists(local_config_path):
        config_path = local_config_path

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config or {}
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def setup_logging(config: Dict[str, Any]):
    """
    配置日志系统
    每次启动时创建一个带时间戳的新日志文件

    Args:
        config: 配置字典
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "logs")
    log_file_prefix = log_config.get("log_file", "ai_companion").replace(".log", "")

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_file_prefix}_{timestamp}.log"

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    # 抑制第三方库的DEBUG日志，减少日志噪音
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("日志系统初始化完成")
    logging.info(f"日志文件: {os.path.join(log_dir, log_file)}")
