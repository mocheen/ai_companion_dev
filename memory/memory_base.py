"""
记忆系统基类
定义记忆系统的接口，具体实现可以后续开发
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from chat.chat_manager import ChatMessage


class MemorySystemBase(ABC):
    """记忆系统基类"""

    @abstractmethod
    def get_short_term_memory(self) -> List[ChatMessage]:
        """
        获取短期记忆（最近对话历史）

        Returns:
            短期记忆消息列表
        """
        pass

    @abstractmethod
    def get_long_term_memory(self, query: str) -> str:
        """
        获取长期记忆（基于查询进行语义匹配）

        Args:
            query: 查询文本

        Returns:
            匹配的长期记忆文本
        """
        pass

    @abstractmethod
    def get_medium_term_memory(self, query: str) -> str:
        """
        获取中期记忆（基于查询进行语义匹配）

        Args:
            query: 查询文本

        Returns:
            匹配的中期记忆文本
        """
        pass

    @abstractmethod
    def add_conversation(self, user_message: str, assistant_message: str):
        """
        添加对话记录到记忆系统

        Args:
            user_message: 用户消息
            assistant_message: AI回复
        """
        pass


class SimpleMemorySystem(MemorySystemBase):
    """
    简单记忆系统实现
    仅保存短期对话历史，不使用向量数据库
    用于初期测试和开发
    """

    def __init__(self, max_turns: int = 12):
        """
        初始化简单记忆系统

        Args:
            max_turns: 最大保存对话轮数
        """
        self.max_turns = max_turns
        self.conversations: List[ChatMessage] = []

    def get_short_term_memory(self) -> List[ChatMessage]:
        """获取短期记忆"""
        return self.conversations.copy()

    def get_long_term_memory(self, query: str) -> str:
        """获取长期记忆（简单版本返回空）"""
        raise NotImplementedError("SimpleMemorySystem 不支持长期记忆功能，请使用 FullMemorySystem")

    def get_medium_term_memory(self, query: str) -> str:
        """获取中期记忆（简单版本返回空）"""
        raise NotImplementedError("SimpleMemorySystem 不支持中期记忆功能，请使用 FullMemorySystem")

    def add_conversation(self, user_message: str, assistant_message: str):
        """
        添加对话记录

        Args:
            user_message: 用户消息
            assistant_message: AI回复
        """
        # 添加用户消息
        self.conversations.append(ChatMessage(role="user", content=user_message))

        # 添加AI回复
        self.conversations.append(ChatMessage(role="assistant", content=assistant_message))

        # 限制记忆长度
        max_messages = self.max_turns * 2  # 每轮对话包含用户和AI两条消息
        if len(self.conversations) > max_messages:
            self.conversations = self.conversations[-max_messages:]
