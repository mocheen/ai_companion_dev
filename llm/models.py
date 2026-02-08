"""
统一的LLM消息和工具调用模型
兼容多种LLM提供商的格式
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    统一的消息模型
    """
    role: Union[str, MessageRole]
    content: str
    tool_call_id: Optional[str] = None  # 工具返回消息需要
    tool_calls: Optional[List['ToolCall']] = None  # 消息包含的工具调用
    timestamp: Optional[str] = None  # 可选的时间戳

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content
        }

        # 工具返回消息（role="tool"）
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        # 包含工具调用的消息（通常是assistant消息）
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        tool_calls = data.get("tool_calls")
        if tool_calls:
            tool_calls = [ToolCall.from_dict(tc) for tc in tool_calls]

        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=tool_calls
        )


@dataclass
class ToolCall:
    """
    统一的工具调用模型
    """
    id: str  # 工具调用的唯一标识
    type: str = "function"  # 目前只支持function类型
    function: 'FunctionCall' = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict() if self.function else {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """从字典创建工具调用"""
        func_data = data.get("function", {})
        return cls(
            id=data["id"],
            type=data.get("type", "function"),
            function=FunctionCall.from_dict(func_data) if func_data else None
        )


@dataclass
class FunctionCall:
    """
    函数调用信息
    """
    name: str  # 函数名称
    arguments: str  # JSON字符串格式的参数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "arguments": self.arguments
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionCall':
        """从字典创建函数调用"""
        return cls(
            name=data["name"],
            arguments=data.get("arguments", "{}")
        )


@dataclass
class Tool:
    """
    工具定义模型（用于发送给LLM）
    """
    type: str = "function"
    function: 'FunctionDefinition' = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type,
            "function": self.function.to_dict() if self.function else {}
        }


@dataclass
class FunctionDefinition:
    """
    函数定义（工具的schema）
    """
    name: str  # 函数名称
    description: str  # 函数描述
    parameters: Dict[str, Any]  # JSON Schema格式的参数定义

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionDefinition':
        """从字典创建函数定义"""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {"type": "object", "properties": {}})
        )


@dataclass
class LLMResponse:
    """
    LLM响应的统一模型
    """
    content: Optional[str] = None  # 文本内容
    tool_calls: Optional[List[ToolCall]] = None  # 工具调用列表
    finish_reason: Optional[str] = None  # 结束原因（stop、tool_calls、length等）
    usage: Optional[Dict[str, int]] = None  # token使用情况

    def has_tool_calls(self) -> bool:
        """检查是否包含工具调用"""
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """转换为消息对象（用于添加到对话历史）"""
        return Message(
            role=MessageRole.ASSISTANT,
            content=self.content or "",
            tool_calls=self.tool_calls
        )


@dataclass
class LLMRequest:
    """
    LLM请求的统一模型
    """
    messages: List[Message]  # 消息列表
    tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None  # 可用工具列表（Tool对象或字典）
    tool_choice: Optional[str] = None  # 工具选择策略（auto、any、none等）
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """转换为API请求格式"""
        result = {
            "messages": [msg.to_dict() for msg in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        if self.tools:
            # 处理工具列表（可能是Tool对象或字典）
            result["tools"] = []
            for tool in self.tools:
                if isinstance(tool, dict):
                    result["tools"].append(tool)
                elif hasattr(tool, 'to_dict'):
                    result["tools"].append(tool.to_dict())
                else:
                    result["tools"].append(tool)

        if self.tool_choice:
            result["tool_choice"] = self.tool_choice

        return result
