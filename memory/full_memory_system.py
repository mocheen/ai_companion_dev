"""
完整的记忆系统实现
包含短期记忆管理、向量数据库存储、Agent工具等
"""

import os
import json
import logging
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from threading import Thread, Lock

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult

from memory.memory_base import MemorySystemBase
from memory.memory_models import (
    ShortTermMessage, MediumTermMemory, LongTermMemory,
    EmotionType, DialogueType, LongTermMemoryType
)
from memory.vector_store import VectorStore
from chat.zhipu_client import ZhipuClient


class ChatMessage:
    """聊天消息类（简化版，避免循环导入）"""

    def __init__(self, role: str, content: str, timestamp):
        self.role = role
        self.content = content
        self.timestamp = timestamp

    def __str__(self) -> str:
        """返回格式化的消息文本"""
        time_str = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "??"
        role_display = "用户" if self.role == "user" else "AI"
        return f"[{time_str}] {role_display}: {self.content}"

logger = logging.getLogger(__name__)


class ZhipuLLM(LLM):
    """智谱API的LangChain LLM封装"""

    client: ZhipuClient = None
    temperature: float = 0.7
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "zhipu"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用LLM"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 记录debug日志
        logger.debug(f"LLM输入: {prompt}")
        logger.debug(f"LLM返回: {response}")

        return response or ""


class FullMemorySystem(MemorySystemBase):
    """
    完整的记忆系统实现
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化记忆系统

        Args:
            config: 配置字典
        """
        self.config = config
        self.lock = Lock()

        # 短期记忆配置
        short_term_config = config.get("short_term_memory", {})
        self.max_turns = short_term_config.get("max_turns", 20)
        self.max_messages = self.max_turns * 2

        # 短期记忆队列
        self.short_term_memory: List[ShortTermMessage] = []

        # 初始化向量数据库
        self.vector_store = VectorStore(config)

        # 初始化LLM客户端（用于Agent）
        api_config = config.get("api", {})
        self.zhipu_client = ZhipuClient(
            api_key=api_config.get("api_key", ""),
            model=api_config.get("model", "glm-4-flash"),
            base_url=api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/")
        )

        # 记忆检索配置
        retrieval_config = config.get("memory_retrieval", {})
        self.medium_term_count = retrieval_config.get("medium_term_count", 3)
        self.long_term_count = retrieval_config.get("long_term_count", 2)

        logger.info("完整记忆系统初始化完成")

    def get_short_term_memory(self) -> List[ChatMessage]:
        """获取短期记忆"""
        with self.lock:
            return [
                ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp
                )
                for msg in self.short_term_memory
            ]

    def get_long_term_memory(self, query: str) -> str:
        """获取长期记忆"""
        if not query:
            return ""

        try:
            memories = self.vector_store.search_long_term_memories(
                query=query,
                n_results=self.long_term_count
            )

            if not memories:
                return ""

            # 格式化为文本
            lines = []
            for i, mem in enumerate(memories, 1):
                data = mem["data"]
                lines.append(f"{i}. {data['topic']}: {data['abstract_summary']}")

            return "\n".join(lines)

        except Exception as e:
            # 向量检索失败时记录错误，但不中断对话
            logger.error(f"获取长期记忆失败: {e}", exc_info=True)
            return ""

    def get_medium_term_memory(self, query: str) -> str:
        """获取中期记忆"""
        if not query:
            return ""

        try:
            memories = self.vector_store.search_medium_term_memories(
                query=query,
                n_results=self.medium_term_count
            )

            if not memories:
                return ""

            # 格式化为文本
            lines = []
            for i, mem in enumerate(memories, 1):
                data = mem["data"]
                lines.append(f"{i}. {data['topic_summary']}")
                lines.append(f"   关键信息: {', '.join(data['key_points'])}")

            return "\n".join(lines)

        except Exception as e:
            # 向量检索失败时记录错误，但不中断对话
            logger.error(f"获取中��记忆失败: {e}", exc_info=True)
            return ""

    def add_conversation(self, user_message: str, assistant_message: str):
        """添加对话记录"""
        with self.lock:
            # 添加用户消息
            self.short_term_memory.append(ShortTermMessage(
                message_id=str(uuid.uuid4()),
                role="user",
                content=user_message
            ))

            # 添加AI回复
            self.short_term_memory.append(ShortTermMessage(
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=assistant_message
            ))

            # 限制记忆长度
            if len(self.short_term_memory) > self.max_messages:
                self.short_term_memory = self.short_term_memory[-self.max_messages:]

            logger.debug(f"添加对话: 当前短期记忆数量 {len(self.short_term_memory)}")

            # 检查是否需要归档
            if len(self.short_term_memory) >= self.max_messages:
                logger.info("短期记忆达到上限，触发归档流程")
                self._trigger_archive_async()

    def _trigger_archive_async(self):
        """异步触发归档流程"""
        def archive_worker():
            try:
                self._archive_short_term_memory()
            except Exception as e:
                logger.error(f"归档短期记忆失败: {e}", exc_info=True)

        thread = Thread(target=archive_worker, daemon=True)
        thread.start()

    def _archive_short_term_memory(self):
        """
        归档短期记忆为中期记忆
        使用LangChain Agent
        """
        logger.info("开始归档短期记忆")

        # 加载Agent提示词
        prompt_path = self.config.get("prompts", {}).get(
            "archive_agent",
            "prompts/archive_agent_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            agent_prompt = f.read().strip()

        # 定义工具
        tools = self._create_archive_tools()

        # 初始化LLM
        llm = ZhipuLLM(client=self.zhipu_client, temperature=0.3)

        # 初始化Agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )

        # 执行Agent
        result = agent.run(agent_prompt)
        logger.info(f"归档Agent执行完成: {result}")

        # Agent执行成功后，根据标记删除已归档的消息
        self._clean_archived_messages()

    def _clean_archived_messages(self):
        """
        根据话题结束标记清理已归档的短期记忆
        保留最后一个话题（即最后一个is_topic_end标记之后的所有消息）
        """
        with self.lock:
            # 找到所有话题结束标记的位置
            topic_end_indices = []
            for i, msg in enumerate(self.short_term_memory):
                if msg.is_topic_end:
                    topic_end_indices.append(i)

            if not topic_end_indices:
                logger.warning("没有找到话题结束标记，不清理短期记忆")
                return

            # 最后一个话题结束的位置
            last_topic_end_index = topic_end_indices[-1]

            # 保留从最后一个话题结束位置开始的所有消息
            original_count = len(self.short_term_memory)
            self.short_term_memory = self.short_term_memory[last_topic_end_index:]

            # 清除所有标记
            for msg in self.short_term_memory:
                msg.is_topic_end = False

            removed_count = original_count - len(self.short_term_memory)
            logger.info(f"清理已归档消息: 删除了{removed_count}条，保留了{len(self.short_term_memory)}条")

    def _create_archive_tools(self) -> List[Tool]:
        """创建归档Agent的工具集"""

        def get_short_term_memory_func() -> str:
            """获取短期记忆"""
            memories = self.get_short_term_memory()
            result = []
            for msg in memories:
                result.append({
                    "id": getattr(msg, 'message_id', 'unknown'),
                    "role": msg.role,
                    "content": msg.content,
                    "time": msg.timestamp.strftime("%H:%M:%S")
                })
            return json.dumps(result, ensure_ascii=False, indent=2)

        def split_topics_func(topic_end_ids: List[str]) -> str:
            """
            标记话题结束位置
            topic_end_ids: 每个话题最后一条消息的ID列表
            会在对应消息上设置is_topic_end标记，不删除任何消息
            """
            with self.lock:
                if not topic_end_ids:
                    return json.dumps({"status": "error", "message": "话题结束ID列表为空"})

                # 清除之前的标记
                for msg in self.short_term_memory:
                    msg.is_topic_end = False

                # 设置新的标记
                marked_count = 0
                for msg_id in topic_end_ids:
                    for msg in self.short_term_memory:
                        if msg.message_id == msg_id:
                            msg.is_topic_end = True
                            marked_count += 1
                            break

                logger.info(f"标记话题结束: 共标记{marked_count}个话题结束位置")

                return json.dumps({
                    "status": "success",
                    "marked_count": marked_count,
                    "topic_end_ids": topic_end_ids
                })

        def create_memory_card_func(cards_json: str) -> str:
            """创建记忆卡片"""
            try:
                cards_data = json.loads(cards_json)

                if not isinstance(cards_data, list):
                    cards_data = [cards_data]

                created_ids = []
                for card_data in cards_data:
                    memory = MediumTermMemory(
                        topic_summary=card_data["topic_summary"],
                        key_points=card_data["key_points"],
                        topic_tags=card_data.get("topic_tags", []),
                        importance_score=card_data.get("importance_score", 0.5),
                        emotion=EmotionType(card_data.get("emotion", "neutral")),
                        dialogue_type=DialogueType(card_data.get("dialogue_type", "casual")),
                        source_message_ids=card_data.get("source_message_ids", [])
                    )

                    memory_id = self.vector_store.add_medium_term_memory(memory)
                    created_ids.append(memory_id)
                    logger.info(f"创建中期记忆: {memory.topic_summary[:50]}...")

                return json.dumps({
                    "status": "success",
                    "created_ids": created_ids,
                    "count": len(created_ids)
                })

            except Exception as e:
                logger.error(f"创建记忆卡片失败: {e}")
                return json.dumps({"status": "error", "message": str(e)})

        return [
            Tool(
                name="get_short_term_memory",
                func=get_short_term_memory_func,
                description="获取完整的短期记忆队列，返回包含消息ID、角色、内容和时间的JSON列表"
            ),
            Tool(
                name="split_topics",
                func=split_topics_func,
                description="根据语义将对话分割为多个话题。传入每个话题最后一条消息的ID列表。系统会在这些消息上设置话题结束标记，不会删除任何消息。等Agent完成所有任务后，程序会自动根据标记清理已归档的���息。"
            ),
            Tool(
                name="create_memory_card",
                func=create_memory_card_func,
                description="为话题创建中期记忆卡片。传入包含记忆卡片字段的JSON或JSON列表。注意：只需要为除最后一个话题外的所有话题创建记忆卡片。"
            )
        ]

    def reorganize_medium_term_memory(self):
        """
        重整中期记忆
        使用LangChain Agent
        """
        logger.info("开始重整中期记忆")

        # 加载Agent提示词
        prompt_path = self.config.get("prompts", {}).get(
            "reorganize_agent",
            "prompts/reorganize_agent_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            agent_prompt = f.read().strip()

        # 定义工具
        tools = self._create_reorganize_tools()

        # 初始化LLM
        llm = ZhipuLLM(client=self.zhipu_client, temperature=0.3)

        # 初始化Agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )

        # 执行Agent
        result = agent.run(agent_prompt)
        logger.info(f"重整完成: {result}")

    def _create_reorganize_tools(self) -> List[Tool]:
        """创建重整Agent的工具集"""

        def get_medium_term_memories_func(
            days: Optional[int] = None,
            older_than_days: Optional[int] = None,
            min_importance: Optional[float] = None,
            max_importance: Optional[float] = None
        ) -> str:
            """获取中期记忆"""
            memories = self.vector_store.get_all_medium_term_memories()

            # 应用过滤条件
            filtered = []
            for mem in memories:
                data = mem["data"]
                created_at = datetime.fromisoformat(data["created_at"])
                importance = data["importance_score"]

                # 时间过滤
                if days is not None:
                    cutoff = datetime.now() - timedelta(days=days)
                    if created_at < cutoff:
                        continue

                # 查询n天之前的记忆
                if older_than_days is not None:
                    cutoff = datetime.now() - timedelta(days=older_than_days)
                    if created_at >= cutoff:
                        continue

                # 重要性过滤
                if min_importance is not None and importance < min_importance:
                    continue
                if max_importance is not None and importance > max_importance:
                    continue

                filtered.append({
                    "id": mem["id"],
                    "data": data
                })

            return json.dumps(filtered, ensure_ascii=False, indent=2)

        def get_long_term_memories_func() -> str:
            """获取长期记忆"""
            memories = self.vector_store.get_all_long_term_memories()
            return json.dumps(memories, ensure_ascii=False, indent=2)

        def merge_similar_memories_func(
            delete_ids: List[str],
            new_card: str
        ) -> str:
            """合并相似记忆"""
            try:
                # 删除旧记忆
                for mem_id in delete_ids:
                    self.vector_store.delete_medium_term_memory(mem_id)

                # 创建新记忆
                card_data = json.loads(new_card)
                memory = MediumTermMemory(
                    topic_summary=card_data["topic_summary"],
                    key_points=card_data["key_points"],
                    topic_tags=card_data.get("topic_tags", []),
                    importance_score=card_data.get("importance_score", 0.5),
                    emotion=EmotionType(card_data.get("emotion", "neutral")),
                    dialogue_type=DialogueType(card_data.get("dialogue_type", "casual"))
                )

                new_id = self.vector_store.add_medium_term_memory(memory)

                logger.info(f"合并记忆: 删除{len(delete_ids)}个，创建新记忆 {new_id}")

                return json.dumps({
                    "status": "success",
                    "new_id": new_id
                })

            except Exception as e:
                logger.error(f"合并记忆失败: {e}")
                return json.dumps({"status": "error", "message": str(e)})

        def create_long_term_memory_func(cards_json: str) -> str:
            """创建长期记忆"""
            try:
                if isinstance(cards_json, str):
                    cards_data = [json.loads(cards_json)]
                else:
                    cards_data = cards_json

                created_ids = []
                for card_data in cards_data:
                    memory = LongTermMemory(
                        topic=card_data["topic"],
                        abstract_summary=card_data["abstract_summary"],
                        importance_score=card_data.get("importance_score", 0.7),
                        memory_type=LongTermMemoryType(card_data.get("memory_type", "knowledge")),
                        confidence_score=card_data.get("confidence_score", 0.5)
                    )

                    memory_id = self.vector_store.add_long_term_memory(memory)
                    created_ids.append(memory_id)
                    logger.info(f"创建长期记忆: {memory.topic[:50]}...")

                return json.dumps({
                    "status": "success",
                    "created_ids": created_ids,
                    "count": len(created_ids)
                })

            except Exception as e:
                logger.error(f"创建长期记忆失败: {e}")
                return json.dumps({"status": "error", "message": str(e)})

        def delete_medium_term_memories_func(memory_ids: List[str]) -> str:
            """删除中期记忆"""
            deleted_count = 0
            for mem_id in memory_ids:
                if self.vector_store.delete_medium_term_memory(mem_id):
                    deleted_count += 1

            logger.info(f"删除中期记忆: {deleted_count}条")

            return json.dumps({
                "status": "success",
                "deleted_count": deleted_count
            })

        return [
            Tool(
                name="get_medium_term_memories",
                func=get_medium_term_memories_func,
                description="获取中期记忆，可选参数: days(最近N天), older_than_days(N天之前), min_importance(最小重要性), max_importance(最大重要性)"
            ),
            Tool(
                name="get_long_term_memories",
                func=get_long_term_memories_func,
                description="获取所有长期记忆"
            ),
            Tool(
                name="merge_similar_memories",
                func=merge_similar_memories,
                description="合并相似的中期记忆，传入delete_ids(要删除的ID列表)和new_card(新记忆的JSON)"
            ),
            Tool(
                name="create_long_term_memory",
                func=create_long_term_memory_func,
                description="创建长期记忆，传入包含记忆卡片字段的JSON或JSON列表"
            ),
            Tool(
                name="delete_medium_term_memories",
                func=delete_medium_term_memories_func,
                description="删除中期记忆，传入记忆ID列表"
            )
        ]
