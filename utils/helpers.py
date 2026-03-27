"""
工具函数
包含配置加载、日志配置等功能
"""

import os
import logging
import yaml
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_env_file():
    """
    加载 .env 文件中的环境变量
    
    从项目根目录查找并加载 .env 文件
    如果 .env 文件不存在，则使用系统环境变量
    """
    # 查找项目根目录（包含 config 文件夹的目录）
    current_dir = Path(__file__).parent
    
    # 向上查找项目根目录
    project_root = current_dir
    while not (project_root / "config").exists():
        project_root = project_root.parent
        if project_root.parent == project_root:  # 已到达根目录
            break
    
    env_file = project_root / ".env"
    
    if env_file.exists():
        # 手动读取并设置环境变量
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
    else:
        # .env 文件不存在时，直接使用系统环境变量
        # 这样可以支持 Docker 容器通过 docker-compose 设置的环境变量
        pass


def replace_env_vars(obj: Any) -> Any:
    """
    递归替换对象中的环境变量占位符
    
    Args:
        obj: 要处理的对象（字典、列表或基本类型）
        
    Returns:
        替换环境变量后的对象
    """
    if isinstance(obj, dict):
        return {key: replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return _replace_env_in_string(obj)
    else:
        return obj


def _replace_env_in_string(text: str) -> str:
    """
    替换字符串中的环境变量占位符
    
    支持格式:
    - ${VAR_NAME} - 环境变量，不存在则返回空字符串
    - ${VAR_NAME:default} - 环境变量，不存在则返回默认值
    
    Args:
        text: 包含环境变量占位符的字符串
        
    Returns:
        替换后的字符串
    """
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        return os.getenv(var_name, default_value)
    
    return re.sub(pattern, replacer, text)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件，支持环境变量替换

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典，失败返回空字典
    """
    # 先加载 .env 文件
    load_env_file()
    
    # 优先使用本地配置文件
    local_config_path = config_path.replace(".yaml", "_local.yaml")
    if os.path.exists(local_config_path):
        config_path = local_config_path

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                # 替换环境变量占位符
                config = replace_env_vars(config)
            return config or {}
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def setup_logging(config: Dict[str, Any]):
    """
    配置日志系统
    支持文件日志开关

    Args:
        config: 配置字典
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "logs")
    log_file_prefix = log_config.get("log_file", "ai_companion").replace(".log", "")
    enable_file = log_config.get("enable_file", True)

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 构建处理器列表
    handlers = [logging.StreamHandler()]

    # 根据配置决定是否添加文件处理器
    if enable_file and log_file_prefix:
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_file_prefix}_{timestamp}.log"

        # 添加文件处理器
        handlers.append(logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8"))

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # 抑制第三方库的DEBUG日志，减少日志噪音
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("日志系统初始化完成")
    if enable_file and log_file_prefix:
        logging.info(f"日志文件: {os.path.join(log_dir, log_file)}")
    else:
        logging.info("文件日志已禁用，仅输出到控制台")
