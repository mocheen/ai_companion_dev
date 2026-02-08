"""
工具定义和执行系统
"""

import json
import logging
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """
    工具定义
    描述Agent可以调用的函数
    """
    name: str  # 工具名称
    description: str  # 工具描述（提供给LLM）
    function: Callable  # 实际的Python函数
    parameters: Dict[str, Any]  # JSON Schema格式的参数定义

    def to_llm_format(self) -> Dict[str, Any]:
        """
        转换为LLM API期望的工具定义格式

        Returns:
            工具定义字典
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def execute(self, arguments: str) -> str:
        """
        执行工具函数

        Args:
            arguments: JSON字符串格式的参数

        Returns:
            执行结果的字符串表示
        """
        try:
            # 解析参数
            args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments

            # 执行函数
            result = self.function(**args_dict)

            # 如果结果不是字符串，转换为JSON字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            logger.debug(f"工具 {self.name} 执行成功")
            return result

        except json.JSONDecodeError as e:
            error_msg = f"参数JSON解析失败: {e}"
            logger.error(f"工具 {self.name} 执行失败: {error_msg}")
            return json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            logger.error(f"工具 {self.name} 执行失败: {error_msg}")
            return json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False)


class ToolExecutor:
    """
    工具执行器
    管理所有可用工具并执行调用
    """

    def __init__(self):
        """初始化工具执行器"""
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        注册工具

        Args:
            tool: 工具对象
        """
        self.tools[tool.name] = tool
        logger.debug(f"已注册工具: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            工具对象，不存在返回None
        """
        return self.tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        """
        获取所有工具

        Returns:
            工具列表
        """
        return list(self.tools.values())

    def get_tools_for_llm(self) -> list[Dict[str, Any]]:
        """
        获取所有工具的LLM格式定义

        Returns:
            工具定义列表
        """
        return [tool.to_llm_format() for tool in self.tools.values()]

    def execute_tool_call(self, tool_name: str, arguments: str) -> str:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments: JSON字符串格式的参数

        Returns:
            执行结果
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            error_msg = f"工具不存在: {tool_name}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False)

        return tool.execute(arguments)
