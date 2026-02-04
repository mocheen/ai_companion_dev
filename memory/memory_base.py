"""
记忆系统基类
定义记忆系统的接口，具体实现可以后续开发
"""

import os
import json
import atexit
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from chat.chat_manager import ChatMessage

logger = logging.getLogger(__name__)


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

    def __init__(self, max_turns: int = 12, persist_file: str = "data/short_term_memory.json"):
        """
        初始化简单记忆系统

        Args:
            max_turns: 最大保存对话轮数
            persist_file: 持久化文件路径
        """
        self.max_turns = max_turns
        self.persist_file = persist_file
        self.conversations: List[ChatMessage] = []

        self._load_short_term_memory()
        atexit.register(self._save_short_term_memory)

    def _load_short_term_memory(self):
        """从文件加载短期记忆"""
        if not os.path.exists(self.persist_file):
            logger.info(f"短期记忆文件不存在: {self.persist_file}")
            return

        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.conversations = [ChatMessage.from_dict(msg) for msg in data]
            logger.info(f"已加载 {len(self.conversations)} 条短期记忆")

        except Exception as e:
            logger.error(f"加载短期记忆失败: {e}", exc_info=True)
            self.conversations = []

    def _save_short_term_memory(self):
        """保存短期记忆到文件"""
        try:
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)

            data = []
            for msg in self.conversations:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                data.append(msg_dict)

            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"已保存 {len(self.conversations)} 条短期记忆到 {self.persist_file}")

        except Exception as e:
            logger.error(f"保存短期记忆失败: {e}", exc_info=True)

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
