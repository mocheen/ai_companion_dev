"""
记忆卡片数据模型
定义中期和长期记忆卡片的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class EmotionType(Enum):
    """情感类型枚举"""
    POSITIVE = "positive"  # 积极
    NEGATIVE = "negative"  # 消极
    NEUTRAL = "neutral"    # 中性


class DialogueType(Enum):
    """对话类型枚举"""
    CASUAL = "casual"         # 闲聊
    QUESTION = "question"     # 问答
    EMOTIONAL = "emotional"   # 情感交流
    TASK = "task"            # 任务执行
    KNOWLEDGE = "knowledge"  # 知识分享


class LongTermMemoryType(Enum):
    """长期记忆类型枚举"""
    PREFERENCE = "preference"      # 用户偏好
    RULE = "rule"                 # 长期遵守的规则
    EVENT = "event"               # 重要事件
    KNOWLEDGE = "knowledge"       # 抽象知识
    CHARACTERISTIC = "characteristic"  # 用户特征


@dataclass
class MediumTermMemory:
    """
    中期记忆卡片
    包含话题摘要、关键信息点、情感色彩等
    """
    topic_summary: str                    # 话题摘要
    key_points: List[str]                 # 关键信息点列表
    topic_tags: List[str] = field(default_factory=list)  # 话题标签
    importance_score: float = 0.5         # 重要性评分 (0-1)
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    emotion: EmotionType = EmotionType.NEUTRAL  # 情感色彩
    dialogue_type: DialogueType = DialogueType.CASUAL  # 对话类型
    source_message_ids: List[str] = field(default_factory=list)  # 源消息ID列表

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "topic_summary": self.topic_summary,
            "key_points": self.key_points,
            "topic_tags": self.topic_tags,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "emotion": self.emotion.value,
            "dialogue_type": self.dialogue_type.value,
            "source_message_ids": self.source_message_ids
        }

    def to_search_text(self) -> str:
        """转换为用于语义检索的文本"""
        text = f"{self.topic_summary}\n"
        text += "关键信息: " + ", ".join(self.key_points) + "\n"
        text += "标签: " + ", ".join(self.topic_tags)
        return text

    @classmethod
    def from_dict(cls, data: dict) -> 'MediumTermMemory':
        """从字典创建实例"""
        return cls(
            topic_summary=data["topic_summary"],
            key_points=data["key_points"],
            topic_tags=data.get("topic_tags", []),
            importance_score=data.get("importance_score", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            emotion=EmotionType(data.get("emotion", "neutral")),
            dialogue_type=DialogueType(data.get("dialogue_type", "casual")),
            source_message_ids=data.get("source_message_ids", [])
        )


@dataclass
class LongTermMemory:
    """
    长期记忆卡片
    包含抽象化知识、用户特征、重要规则等
    """
    topic: str                              # 主题
    abstract_summary: str                   # 抽象化知识摘要
    importance_score: float = 0.7           # 重要性评分 (0-1)
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    source_memory_ids: List[str] = field(default_factory=list)  # 源记忆关联
    confidence_score: float = 0.5           # 置信度评分 (0-1)
    memory_type: LongTermMemoryType = LongTermMemoryType.KNOWLEDGE  # 记忆类别

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "topic": self.topic,
            "abstract_summary": self.abstract_summary,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "source_memory_ids": self.source_memory_ids,
            "confidence_score": self.confidence_score,
            "memory_type": self.memory_type.value
        }

    def to_search_text(self) -> str:
        """转换为用于语义检索的文本"""
        text = f"{self.topic}: {self.abstract_summary}"
        return text

    @classmethod
    def from_dict(cls, data: dict) -> 'LongTermMemory':
        """从字典创建实例"""
        return cls(
            topic=data["topic"],
            abstract_summary=data["abstract_summary"],
            importance_score=data.get("importance_score", 0.7),
            created_at=datetime.fromisoformat(data["created_at"]),
            source_memory_ids=data.get("source_memory_ids", []),
            confidence_score=data.get("confidence_score", 0.5),
            memory_type=LongTermMemoryType(data.get("memory_type", "knowledge"))
        )


@dataclass
class ShortTermMessage:
    """
    短期记忆消息
    用于存储在队列中的临时消息
    """
    message_id: str                         # 消息唯一ID
    role: str                               # 角色 ("user" 或 "assistant")
    content: str                            # 消息内容
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    is_topic_end: bool = False              # 是否为话题结束标记

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "is_topic_end": self.is_topic_end
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ShortTermMessage':
        """从字典创建实例"""
        return cls(
            message_id=data["message_id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_topic_end=data.get("is_topic_end", False)
        )
