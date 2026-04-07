"""
配置管理器模块
负责配置的读取、写入、重载和组件重建
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config_items import config_registry

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器

    负责：
    - 读取和写入 config.yaml
    - 获取当前配置值（与 ConfigItem 定义对应）
    - 更新配置到内存
    - 持久化配置到文件
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}

    def _get_nested_value(self, data: dict, key_path: str) -> Any:
        """根据点分隔的 key 路径获取嵌套值

        Args:
            data: 字典数据
            key_path: 如 "chat.temperature"

        Returns:
            对应的值，找不到返回 None
        """
        keys = key_path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _set_nested_value(self, data: dict, key_path: str, value: Any):
        """根据点分隔的 key 路径设置嵌套值

        Args:
            data: 字典数据（会被就地修改）
            key_path: 如 "chat.temperature"
            value: 要设置的值
        """
        keys = key_path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def load_raw_config(self) -> Dict[str, Any]:
        """从 config.yaml 读取原始配置（不替换环境变量）

        Returns:
            原始配置字典
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"读取配置文件失败: {e}")
            return {}

    def save_raw_config(self, config: Dict[str, Any]) -> bool:
        """将配置写入 config.yaml

        Args:
            config: 要写入的配置字典

        Returns:
            是否成功
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"配置已保存到 {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False

    def get_current_values(self, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据注册的 ConfigItem 列表，从已解析的配置中提取当前值

        Args:
            resolved_config: 经过环境变量替换后的配置字典

        Returns:
            {key: value} 的字典
        """
        values = {}
        for item in config_registry.get_all():
            val = self._get_nested_value(resolved_config, item.key)
            if val is not None:
                values[item.key] = val
            else:
                values[item.key] = item.default
        return values

    def update_config(
        self,
        updates: Dict[str, Any],
        current_resolved_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """更新配置：写入文件并更新内存中的配置

        Args:
            updates: 用户提交的 {key: value} 更新
            current_resolved_config: 当前已解析的配置（环境变量替换后的）

        Returns:
            更新后的已解析配置字典
        """
        # 1. 读取原始 yaml（保留格式和未注册的配置项）
        raw_config = self.load_raw_config()

        # 2. 应用更新到原始配置
        for key, value in updates.items():
            item = config_registry.get(key)
            if item is None:
                logger.warning(f"未注册的配置项: {key}，跳过")
                continue
            self._set_nested_value(raw_config, key, value)

        # 3. 写入文件
        if not self.save_raw_config(raw_config):
            raise RuntimeError("保存配置文件失败")

        # 4. 重新加载配置（会重新读取 .env 和环境变量）
        from utils.helpers import load_config
        new_config = load_config(str(self.config_path))

        return new_config

    def get_schema(self) -> List[dict]:
        """获取配置项 schema 列表"""
        return config_registry.get_schema_list()

    def check_rebuild_needed(self, updates: Dict[str, Any]) -> bool:
        """检查更新是否需要重建组件

        Args:
            updates: 用户提交的 {key: value} 更新

        Returns:
            是否需要重建
        """
        for key in updates:
            item = config_registry.get(key)
            if item and item.requires_rebuild:
                return True
        return False
