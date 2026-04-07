"""
配置项定义模块
提供通用的配置项类，支持多种类型，方便扩展
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict


@dataclass
class ConfigItem:
    """通用配置项定义

    Attributes:
        key: 配置键路径，对应 yaml 中的路径，如 "chat.temperature"
        label: 显示名称
        type: 控件类型："slider" | "number" | "string" | "select" | "password"
        group: 分组名称，如 "对话设置"
        default: 默认值
        description: 配置说明（可选）
        min_val: 最小值（slider/number 类型）
        max_val: 最大值（slider/number 类型）
        step: 步长（slider 类型）
        options: 选项列表（select 类型），格式 [{"label": "显示名", "value": "值"}]
        secret: 是否为敏感信息（如 API Key），前端渲染为 password 输入框
        requires_rebuild: 修改后是否需要重建组件
    """

    key: str
    label: str
    type: str  # "slider" | "number" | "string" | "select" | "password"
    group: str
    default: Any = None
    description: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, str]]] = None
    secret: bool = False
    requires_rebuild: bool = False

    def to_schema(self) -> dict:
        """转换为前端可用的 schema 字典"""
        schema = {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "group": self.group,
            "default": self.default,
            "description": self.description,
            "secret": self.secret,
            "requiresRebuild": self.requires_rebuild,
        }
        if self.type in ("slider", "number"):
            if self.min_val is not None:
                schema["min"] = self.min_val
            if self.max_val is not None:
                schema["max"] = self.max_val
            if self.step is not None:
                schema["step"] = self.step
        if self.type == "select" and self.options:
            schema["options"] = self.options
        return schema


class ConfigRegistry:
    """配置项注册表，管理所有可编辑的配置项"""

    def __init__(self):
        self._items: Dict[str, ConfigItem] = {}

    def register(self, item: ConfigItem):
        """注册一个配置项"""
        self._items[item.key] = item

    def get(self, key: str) -> Optional[ConfigItem]:
        """获取指定配置项"""
        return self._items.get(key)

    def get_all(self) -> List[ConfigItem]:
        """获取所有配置项，按 group 分组排序"""
        return list(self._items.values())

    def get_groups(self) -> Dict[str, List[ConfigItem]]:
        """按分组获取配置项"""
        groups: Dict[str, List[ConfigItem]] = {}
        for item in self._items.values():
            if item.group not in groups:
                groups[item.group] = []
            groups[item.group].append(item)
        return groups

    def get_schema_list(self) -> List[dict]:
        """获取所有配置项的 schema 列表"""
        groups = self.get_groups()
        result = []
        for group_name, items in groups.items():
            result.append({
                "group": group_name,
                "items": [item.to_schema() for item in items]
            })
        return result


# 全局配置项注册表实例
config_registry = ConfigRegistry()


def _register_all_items():
    """注册所有可编辑的配置项"""

    # ===== API 配置 =====
    config_registry.register(ConfigItem(
        key="api.api_key",
        label="API Key",
        type="password",
        group="API 配置",
        default="",
        description="主 API 密钥",
        secret=True,
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="api.model",
        label="模型名称",
        type="select",
        group="API 配置",
        default="glm-4.7",
        options=[
            {"label": "GLM-4.7", "value": "glm-4.7"},
            {"label": "GLM-4-Flash", "value": "glm-4-flash"},
            {"label": "GLM-4-Plus", "value": "glm-4-plus"},
            {"label": "GLM-4-Air", "value": "glm-4-air"},
        ],
        description="对话使用的模型",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="api.base_url",
        label="API 地址",
        type="string",
        group="API 配置",
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="API 基础 URL",
        requires_rebuild=True,
    ))

    # ===== Agent API 配置 =====
    config_registry.register(ConfigItem(
        key="agent_api.api_key",
        label="Agent API Key",
        type="password",
        group="Agent 配置",
        default="",
        description="Agent 专用的 API 密钥（用于记忆归档和重整）",
        secret=True,
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="agent_api.model",
        label="Agent 模型",
        type="select",
        group="Agent 配置",
        default="glm-4.7",
        options=[
            {"label": "GLM-4.7", "value": "glm-4.7"},
            {"label": "GLM-4-Flash", "value": "glm-4-flash"},
            {"label": "GLM-4-Plus", "value": "glm-4-plus"},
            {"label": "GLM-4-Air", "value": "glm-4-air"},
        ],
        description="Agent 使用的模型",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="agent_api.base_url",
        label="Agent API 地址",
        type="string",
        group="Agent 配置",
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="Agent 使用的 API 基础 URL",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="agent_api.tool_choice",
        label="工具选择策略",
        type="select",
        group="Agent 配置",
        default="auto",
        options=[
            {"label": "自动", "value": "auto"},
            {"label": "必须使用工具", "value": "required"},
            {"label": "不使用工具", "value": "none"},
        ],
        description="Agent 调用工具的策略",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="agent.archive.max_iterations",
        label="归档最大迭代次数",
        type="number",
        group="Agent 配置",
        default=20,
        min_val=1,
        max_val=100,
        description="归档 Agent 单次执行的最大迭代次数",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="agent.reorganize.max_iterations",
        label="重整最大迭代次数",
        type="number",
        group="Agent 配置",
        default=40,
        min_val=1,
        max_val=100,
        description="重整 Agent 单次执行的最大迭代次数",
        requires_rebuild=True,
    ))

    # ===== 对话设置 =====
    config_registry.register(ConfigItem(
        key="chat.temperature",
        label="温度参数",
        type="slider",
        group="对话设置",
        default=0.7,
        min_val=0,
        max_val=2,
        step=0.1,
        description="控制回复的随机性，值越高越随机",
    ))
    config_registry.register(ConfigItem(
        key="chat.max_tokens",
        label="最大 Token 数",
        type="number",
        group="对话设置",
        default=2000,
        min_val=100,
        max_val=8000,
        description="单次回复的最大 token 数",
    ))
    config_registry.register(ConfigItem(
        key="chat.timeout",
        label="超时时间（秒）",
        type="number",
        group="对话设置",
        default=60,
        min_val=10,
        max_val=300,
        description="API 请求超时时间",
    ))

    # ===== 记忆设置 =====
    config_registry.register(ConfigItem(
        key="short_term_memory.max_turns",
        label="短期记忆上限",
        type="number",
        group="记忆设置",
        default=20,
        min_val=5,
        max_val=50,
        description="触发记忆归档的最大对话轮数",
    ))
    config_registry.register(ConfigItem(
        key="memory_retrieval.medium_term_count",
        label="中期记忆检索数量",
        type="number",
        group="记忆设置",
        default=3,
        min_val=1,
        max_val=10,
        description="检索的中期记忆卡片数量",
    ))
    config_registry.register(ConfigItem(
        key="memory_retrieval.long_term_count",
        label="长期记忆检索数量",
        type="number",
        group="记忆设置",
        default=2,
        min_val=1,
        max_val=10,
        description="检索的长期记忆卡片数量",
    ))
    config_registry.register(ConfigItem(
        key="memory_system.type",
        label="记忆系统类型",
        type="select",
        group="记忆设置",
        default="full",
        options=[
            {"label": "完整记忆", "value": "full"},
            {"label": "简单记忆", "value": "simple"},
        ],
        description="simple 仅使用短期记忆，full 使用完整三层记忆系统",
        requires_rebuild=True,
    ))

    # ===== 向量数据库设置 =====
    config_registry.register(ConfigItem(
        key="vector_db.mode",
        label="数据库模式",
        type="select",
        group="向量数据库",
        default="local",
        options=[
            {"label": "本地模式", "value": "local"},
            {"label": "远程模式", "value": "remote"},
        ],
        description="向量数据库运行模式（修改后将覆盖环境变量值）",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="vector_db.embedding_type",
        label="嵌入方式",
        type="select",
        group="向量数据库",
        default="api",
        options=[
            {"label": "API 嵌入", "value": "api"},
            {"label": "本地嵌入", "value": "local"},
        ],
        description="文本嵌入方式：使用远程 API 或本地模型（修改后将覆盖环境变量值）",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="vector_db.embedding_model",
        label="Embedding 模型",
        type="string",
        group="向量数据库",
        default="embedding-3",
        description="API 嵌入模型名称",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="vector_db.embedding_api_key",
        label="Embedding API Key",
        type="password",
        group="向量数据库",
        default="",
        description="嵌入服务的 API 密钥",
        secret=True,
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="vector_db.embedding_base_url",
        label="Embedding API 地址",
        type="string",
        group="向量数据库",
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="嵌入服务的 API 基础 URL",
        requires_rebuild=True,
    ))
    config_registry.register(ConfigItem(
        key="vector_db.local_embedding_model",
        label="本地 Embedding 模型",
        type="string",
        group="向量数据库",
        default="shibing624/text2vec-base-chinese",
        description="本地嵌入使用的模型名称（HuggingFace 格式）",
        requires_rebuild=True,
    ))

    # ===== 回溯设置 =====
    config_registry.register(ConfigItem(
        key="rollback.enabled",
        label="启用回溯",
        type="select",
        group="回溯设置",
        default="true",
        options=[
            {"label": "启用", "value": "true"},
            {"label": "禁用", "value": "false"},
        ],
        description="是否启用记忆操作的回溯记录功能",
    ))
    config_registry.register(ConfigItem(
        key="rollback.max_snapshots",
        label="最大快照数量",
        type="number",
        group="回溯设置",
        default=50,
        min_val=5,
        max_val=500,
        description="保存的最大回溯快照数量，超过后自动删除最旧的",
    ))
    config_registry.register(ConfigItem(
        key="rollback.storage_dir",
        label="存储目录",
        type="string",
        group="回溯设置",
        default="data/rollback_snapshots",
        description="回溯快照JSON文件的存储目录",
    ))

    # ===== 日志设置 =====
    config_registry.register(ConfigItem(
        key="logging.level",
        label="日志级别",
        type="select",
        group="日志设置",
        default="DEBUG",
        options=[
            {"label": "DEBUG", "value": "DEBUG"},
            {"label": "INFO", "value": "INFO"},
            {"label": "WARNING", "value": "WARNING"},
            {"label": "ERROR", "value": "ERROR"},
        ],
        description="日志输出级别",
    ))
    config_registry.register(ConfigItem(
        key="logging.enable_file",
        label="启用文件日志",
        type="select",
        group="日志设置",
        default="false",
        options=[
            {"label": "启用", "value": "true"},
            {"label": "禁用", "value": "false"},
        ],
        description="是否将日志输出到文件",
    ))

    # ===== 记忆流设置 =====
    config_registry.register(ConfigItem(
        key="memory_flow.archive_interval_days",
        label="归档间隔天数",
        type="number",
        group="记忆流设置",
        default=7,
        min_val=1,
        max_val=90,
        description="自动重整记忆的时间间隔（天）",
    ))

    # ===== 命令设置 =====
    config_registry.register(ConfigItem(
        key="commands.prefix",
        label="命令前缀",
        type="string",
        group="命令设置",
        default="/",
        description="系统命令的触发前缀字符",
    ))


# 模块加载时自动注册所有配置项
_register_all_items()
