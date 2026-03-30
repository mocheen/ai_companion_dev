"""
完整的记忆系统实现
包含短期记忆管理、向量数据库存储、自主Agent归档等
完全脱离LangChain，使用自主Agent框架
"""

import os
import json
import logging
import uuid
import atexit
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from threading import Thread, Lock

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from memory.memory_base import MemorySystemBase
from memory.memory_models import (
    ShortTermMessage, MediumTermMemory, LongTermMemory,
    EmotionType, DialogueType, LongTermMemoryType
)
from memory.vector_store import VectorStore
from llm.zhipu_llm import ZhipuLLM
from llm.models import Message, MessageRole
from agent.base_agent import BaseAgent
from agent.tool import Tool, ToolExecutor


class ChatMessage:
    """聊天消息类（简化版，避免循环导入）"""

    def __init__(self, role: str, content: str, timestamp=None, message_id: str = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.message_id = message_id

    def __str__(self) -> str:
        """返回格式化的消息文本"""
        time_str = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "??"
        role_display = "用户" if self.role == "user" else "AI"
        return f"[{time_str}] {role_display}: {self.content}"

logger = logging.getLogger(__name__)


class FullMemorySystem(MemorySystemBase):
    """
    完整的记忆系统实现

    使用自主Agent框架实现记忆归档和重整
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
        self.max_messages = short_term_config.get("max_turns", 20)

        # 初始化Redis
        redis_config = config.get("redis", {})
        self.redis_prefix = redis_config.get("prefix", "ai_companion:")
        self.redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password", "") if redis_config.get("password") else None,
            decode_responses=True
        )

        # 尝试修复Redis RDB快照错误问题
        try:
            self.redis_client.config_set("stop-writes-on-bgsave-error", "no")
            logger.info("已临时禁用Redis RDB快照错误检查")
        except Exception as e:
            logger.warning(f"无法自动修复Redis配置: {e}")

        # Redis键名
        self.redis_key_short_term = f"{self.redis_prefix}short_term_memory"
        self.redis_key_flow_state = f"{self.redis_prefix}flow_state"

        # 初始化向量数据库
        self.vector_store = VectorStore(config)

        # 初始化LLM客户端（用于普通对话）
        api_config = config.get("api", {})
        self.llm_client = ZhipuLLM(
            api_key=api_config.get("api_key", ""),
            model=api_config.get("model", "glm-4-flash"),
            base_url=api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/")
        )

        # 初始化Agent专用的LLM客户端（用于记忆归档和重整）
        agent_api_config = config.get("agent_api", {})
        if not agent_api_config:
            logger.warning("未配置agent_api，将使用普通api配置")
            agent_api_config = api_config

        self.agent_llm = ZhipuLLM(
            api_key=agent_api_config.get("api_key", api_config.get("api_key", "")),
            model=agent_api_config.get("model", api_config.get("model", "glm-4-flash")),
            base_url=agent_api_config.get("base_url", api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/"))
        )
        self.agent_tool_choice = agent_api_config.get("tool_choice", "auto")

        # 记忆检索配置
        retrieval_config = config.get("memory_retrieval", {})
        self.medium_term_count = retrieval_config.get("medium_term_count", 3)
        self.long_term_count = retrieval_config.get("long_term_count", 2)

        # 记忆流转配置
        memory_flow_config = config.get("memory_flow", {})
        self.reorganize_interval_days = memory_flow_config.get("archive_interval_days", 7)

        # 加载短期记忆
        self.short_term_memory: List[ShortTermMessage] = self._load_short_term_memory()

        # 检查是否需要归档（启动时）
        if len(self.short_term_memory) >= self.max_messages:
            logger.info(f"短期记忆已达到上限（{len(self.short_term_memory)}/{self.max_messages}），触发归档流程")
            self._trigger_archive_async()

        # 检查是否需要重整中期记忆（启动时）
        if self._should_reorganize():
            logger.info("启动时检测到需要重整中期记忆，触发重整流程")
            self._trigger_reorganize_async()

        logger.info("完整记忆系统初始化完成")

    def _load_short_term_memory(self) -> List[ShortTermMessage]:
        """从Redis加载短期记忆"""
        try:
            data_json = self.redis_client.get(self.redis_key_short_term)
            if data_json:
                data = json.loads(data_json)
                memories = [ShortTermMessage.from_dict(msg) for msg in data]
                logger.info(f"已从Redis加载 {len(memories)} 条短期记忆")
                return memories
            else:
                logger.info("Redis中无短期记忆数据")
                return []
        except Exception as e:
            logger.error(f"从Redis加载短期记忆失败: {e}", exc_info=True)
            return []

    def _save_short_term_memory(self):
        """保存短期记忆到Redis"""
        try:
            data = [msg.to_dict() for msg in self.short_term_memory]
            self.redis_client.set(self.redis_key_short_term, json.dumps(data, ensure_ascii=False))
            logger.debug(f"已保存 {len(self.short_term_memory)} 条短期记忆到Redis")
        except Exception as e:
            logger.error(f"保存短期记忆到Redis失败: {e}", exc_info=True)

    def _load_flow_state(self) -> Dict[str, Any]:
        """从Redis加载记忆流转状态"""
        default_state = {"last_reorganize_time": None}
        try:
            data_json = self.redis_client.get(self.redis_key_flow_state)
            if data_json:
                data = json.loads(data_json)
                return data
            else:
                return default_state
        except Exception as e:
            logger.error(f"从Redis加载记忆流转状态失败: {e}", exc_info=True)
            return default_state

    def _save_flow_state(self, state: Dict[str, Any]):
        """保存记忆流转状态到Redis"""
        try:
            self.redis_client.set(self.redis_key_flow_state, json.dumps(state, ensure_ascii=False))
            logger.debug("已保存记忆流转状态到Redis")
        except Exception as e:
            logger.error(f"保存记忆流转状态到Redis失败: {e}", exc_info=True)

    def _should_reorganize(self) -> bool:
        """
        检查是否需要触发中期记忆重整

        条件：从未重整过 或 距离上次重整超过配置的天数
        """
        state = self._load_flow_state()
        last_time = state.get("last_reorganize_time")

        if not last_time:
            logger.info("从未进行过中期记忆重整，需要执行")
            return True

        try:
            last_dt = datetime.fromisoformat(last_time)
            elapsed = datetime.now() - last_dt

            if elapsed >= timedelta(days=self.reorganize_interval_days):
                logger.info(
                    f"距离上次重整已过 {elapsed.days} 天，超过阈值 {self.reorganize_interval_days} 天，需要执行"
                )
                return True
            else:
                logger.debug(
                    f"距离上次重整仅过 {elapsed.days} 天，未超过阈值 {self.reorganize_interval_days} 天，暂不执行"
                )
                return False
        except Exception as e:
            logger.error(f"解析上次重整时间失败: {e}")
            return True

    def _update_reorganize_time(self):
        """更新上次重整时间"""
        state = self._load_flow_state()
        state["last_reorganize_time"] = datetime.now().isoformat()
        self._save_flow_state(state)
        logger.info(f"已更新上次重整时间: {state['last_reorganize_time']}")

    def get_short_term_memory(self) -> List[ChatMessage]:
        """获取短期记忆（从Redis重新加载）"""
        self.short_term_memory = self._load_short_term_memory()
        with self.lock:
            return [
                ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    message_id=msg.message_id
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
            logger.error(f"获取中期记忆失败: {e}", exc_info=True)
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

            # 保存到Redis
            self._save_short_term_memory()

            # 检查是否需要归档
            if len(self.short_term_memory) >= self.max_messages:
                logger.info("短期记忆达到上限，触发归档流程")
                self._trigger_archive_async()

            # 检查是否需要重整中期记忆
            if self._should_reorganize():
                logger.info("对话后检测到需要重整中期记忆，触发重整流程")
                self._trigger_reorganize_async()

        # 锁释放后通知状态更新
        self._notify_status_update()

    def _trigger_archive_async(self):
        """异步触发归档流程"""
        def archive_worker():
            try:
                self._archive_short_term_memory()
            except Exception as e:
                logger.error(f"归档短期记忆失败: {e}", exc_info=True)

        thread = Thread(target=archive_worker, daemon=True)
        thread.start()

    def _trigger_reorganize_async(self):
        """异步触发中期记忆重整流程"""
        def reorganize_worker():
            try:
                self.reorganize_medium_term_memory()
                # 重整成功后更新时间
                self._update_reorganize_time()
            except Exception as e:
                logger.error(f"重整中期记忆失败: {e}", exc_info=True)

        thread = Thread(target=reorganize_worker, daemon=True)
        thread.start()

    def _archive_short_term_memory(self):
        """
        归档短期记忆为中期记忆
        使用自主Agent框架
        """
        logger.info("开始归档短期记忆")

        # 加载Agent提示词
        prompt_path = self.config.get("prompts", {}).get(
            "archive_agent",
            "prompts/archive_agent_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            agent_prompt = f.read().strip()

        # 创建工具执行器
        tool_executor = ToolExecutor()

        # 注册归档工具
        self._register_archive_tools(tool_executor)

        # 从配置获取归档Agent最大迭代次数
        agent_config = self.config.get("agent", {})
        archive_config = agent_config.get("archive", {})
        archive_max_iterations = archive_config.get("max_iterations", 10)

        # 创建Agent
        agent = BaseAgent(
            llm=self.agent_llm,
            tool_executor=tool_executor,
            system_prompt=agent_prompt,
            max_iterations=archive_max_iterations
        )

        logger.info(f"初始化归档Agent，使用模型: {self.agent_llm.model}, tool_choice: {self.agent_tool_choice}, max_iterations: {archive_max_iterations}")

        # 带重试机制的执行
        max_retries = 3
        base_delay = 5  # 秒
        archive_timeout = 180  # 归档任务超时时间（3分钟）

        for attempt in range(max_retries):
            try:
                result = agent.run("请开始归档短期记忆", tool_choice=self.agent_tool_choice, timeout=archive_timeout)
                logger.info(f"归档Agent执行完成: {result}")
                break  # 成功，跳出重试循环

            except Exception as e:
                # 判断是否为速率限制错误或超时错误
                error_str = str(e)
                is_rate_limit = "429" in error_str or "Too Many Requests" in error_str
                is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

                if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                    # 指数退避：5秒、10秒、20秒
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"遇到API{'速率限制' if is_rate_limit else '超时'}（第{attempt + 1}次尝试），"
                        f"{delay}秒后重试... 错误: {e}"
                    )
                    time.sleep(delay)
                else:
                    # 非速率限制/超时错误，或已达最大重试次数
                    logger.error(f"归档失败: {e}")
                    raise

        # Agent执行成功后，根据标记删除已归档的消息
        self._clean_archived_messages()

    def _register_archive_tools(self, tool_executor: ToolExecutor):
        """注册归档Agent的工具集"""

        def get_short_term_memory_func() -> str:
            """获取短期记忆（限制数量以避免超时）"""
            memories = self.get_short_term_memory()
            
            # 限制返回的记忆数量，避免一次性发送太多内容导致超时
            # 最多返回最近50条记忆，如果超过则只返回最近的
            max_memories = 50
            if len(memories) > max_memories:
                logger.warning(f"短期记忆数量({len(memories)})超过限制({max_memories})，只返回最近的{max_memories}条")
                memories = memories[-max_memories:]
            
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
            """标记话题结束位置"""
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

                # 保存标记到Redis，防止服务重启导致标记丢失
                self._save_short_term_memory()

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
                if len(cards_data) > 1:
                    # 使用批量处理，避免触发速率限制
                    memories = []
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
                        memories.append(memory)
                    
                    created_ids = self.vector_store.add_medium_term_memories_batch(memories)
                    for memory in memories:
                        logger.info(f"创建中期记忆: {memory.topic_summary[:50]}...")
                else:
                    # 单个记忆，使用普通方法
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

        # 注册工具
        tool_executor.register_tool(Tool(
            name="get_short_term_memory",
            description="获取完整的短期记忆队列，返回包含消息ID、角色、内容和时间的JSON列表",
            function=get_short_term_memory_func,
            parameters={"type": "object", "properties": {}, "required": []}
        ))

        tool_executor.register_tool(Tool(
            name="split_topics",
            description="根据语义将对话分割为多个话题。传入需要归档的话题最后一条消息的ID列表（不要为最后一个话题标记！）。系统会在这些消息上设置话题结束标记，等Agent完成所有任务后，程序会自动删除标记之前的已归档消息，只保留最后一个可能未结束的话题。",
            function=split_topics_func,
            parameters={
                "type": "object",
                "properties": {
                    "topic_end_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要归档的话题最后一条消息的ID列表（不要包含最后一个话题！）"
                    }
                },
                "required": ["topic_end_ids"]
            }
        ))

        tool_executor.register_tool(Tool(
            name="create_memory_card",
            description="为话题创建中期记忆卡片。传入包含记忆卡片字段的JSON或JSON列表。注意：只需要为除最后一个话题外的所有话题创建记忆卡片。",
            function=create_memory_card_func,
            parameters={
                "type": "object",
                "properties": {
                    "cards_json": {
                        "type": "string",
                        "description": "包含记忆卡片字段的JSON字符串或JSON列表字符串"
                    }
                },
                "required": ["cards_json"]
            }
        ))

        logger.debug("归档工具注册完成")

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

            # 保存清理后的短期记忆
            self._save_short_term_memory()

    def reorganize_medium_term_memory(self):
        """
        重整中期记忆
        使用自主Agent框架
        """
        logger.info("开始重整中期记忆")

        # 加载Agent提示词
        prompt_path = self.config.get("prompts", {}).get(
            "reorganize_agent",
            "prompts/reorganize_agent_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            agent_prompt = f.read().strip()

        # 注入时间信息到prompt
        state = self._load_flow_state()
        last_reorganize_time = state.get("last_reorganize_time")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_reorganize_display = last_reorganize_time if last_reorganize_time else "从未重整过"

        agent_prompt = agent_prompt.replace("{current_time}", current_time)
        agent_prompt = agent_prompt.replace("{last_reorganize_time}", last_reorganize_display)
        logger.debug(f"注入时间信息 - 当前时间: {current_time}, 上次重整: {last_reorganize_display}")

        # 创建工具执行器
        tool_executor = ToolExecutor()

        # 注册重整工具
        self._register_reorganize_tools(tool_executor)

        # 从配置获取重整Agent最大迭代次数
        agent_config = self.config.get("agent", {})
        reorganize_config = agent_config.get("reorganize", {})
        reorganize_max_iterations = reorganize_config.get("max_iterations", 15)

        # 创建Agent
        agent = BaseAgent(
            llm=self.agent_llm,
            tool_executor=tool_executor,
            system_prompt=agent_prompt,
            max_iterations=reorganize_max_iterations
        )

        logger.info(f"初始化重整Agent，使用模型: {self.agent_llm.model}, tool_choice: {self.agent_tool_choice}, max_iterations: {reorganize_max_iterations}")

        # 带重试机制的执行（与归档Agent保持一致）
        max_retries = 3
        base_delay = 5  # 秒
        reorganize_timeout = 300  # 重整任务超时时间（5分钟，比归档更长）

        for attempt in range(max_retries):
            try:
                result = agent.run("请开始重整中期记忆", tool_choice=self.agent_tool_choice, timeout=reorganize_timeout)
                logger.info(f"重整Agent执行完成: {result}")
                break  # 成功，跳出重试循环

            except Exception as e:
                # 判断是否为速率限制错误或超时错误
                error_str = str(e)
                is_rate_limit = "429" in error_str or "Too Many Requests" in error_str
                is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

                if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                    # 指数退避：5秒、10秒、20秒
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"遇到API{'速率限制' if is_rate_limit else '超时'}（第{attempt + 1}次尝试），"
                        f"{delay}秒后重试... 错误: {e}"
                    )
                    time.sleep(delay)
                else:
                    # 非速率限制/超时错误，或已达最大重试次数
                    logger.error(f"重整失败: {e}")
                    raise

        # TODO: 可以添加重整后的验证逻辑
        # self._verify_reorganization_result()
        logger.info("中期记忆重整流程完成")

    def _register_reorganize_tools(self, tool_executor: ToolExecutor):
        """注册重整Agent的工具集"""

        def get_medium_term_memories_func(
            days: Optional[int] = None,
            older_than_days: Optional[int] = None,
            min_importance: Optional[float] = None,
            max_importance: Optional[float] = None,
            limit: Optional[int] = None
        ) -> str:
            """
            获取中期记忆

            Args:
                days: 获取最近N天创建的记忆
                older_than_days: 获取N天之前创建的记忆
                min_importance: 最小重要性评分
                max_importance: 最大重要性评分
                limit: 最多返回的记忆数量（默认100，避免超时）
            """
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

            # 应用数量限制（避免返回太多数据导致超时）
            max_memories = limit or 100  # 默认最多100条
            if len(filtered) > max_memories:
                logger.warning(f"中期记忆过滤后数量({len(filtered)})超过限制({max_memories})，只返回最近的{max_memories}条")
                filtered = filtered[:max_memories]

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

        # 注册工具
        tool_executor.register_tool(Tool(
            name="get_medium_term_memories",
            description=(
                "获取中期记忆卡片。支持多种筛选条件："
                "days(最近N天创建)、older_than_days(N天之前创建)、"
                "min_importance(最小重要性)、max_importance(最大重要性)、"
                "limit(最多返回条数，默认100)。"
                "建议先不加筛选条件获取总数，再根据需要使用筛选条件。"
            ),
            function=get_medium_term_memories_func,
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "number",
                        "description": "获取最近N天创建的记忆"
                    },
                    "older_than_days": {
                        "type": "number",
                        "description": "获取N天之前创建的记忆"
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "最小重要性评分(0-1)"
                    },
                    "max_importance": {
                        "type": "number",
                        "description": "最大重要性评分(0-1)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "最多返回的记忆数量（默认100，最大建议200）"
                    }
                },
                "required": []
            }
        ))

        tool_executor.register_tool(Tool(
            name="get_long_term_memories",
            description="获取所有长期记忆卡片。用于查看已建立的长期知识，避免重复创建。",
            function=get_long_term_memories_func,
            parameters={"type": "object", "properties": {}, "required": []}
        ))

        tool_executor.register_tool(Tool(
            name="merge_similar_memories",
            description=(
                "合并相似的中期记忆卡片。"
                "传入delete_ids（要删除的旧记忆ID列表）和new_card（新记忆的JSON字符串）。"
                "用于将内容相似的记忆合并为一个更完整的记忆。"
                "注意：合并后的新记忆会保留在记忆系统中。"
            ),
            function=merge_similar_memories_func,
            parameters={
                "type": "object",
                "properties": {
                    "delete_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要删除的旧记忆ID列表"
                    },
                    "new_card": {
                        "type": "string",
                        "description": "新记忆的JSON字符串，包含topic_summary, key_points等字段"
                    }
                },
                "required": ["delete_ids", "new_card"]
            }
        ))

        tool_executor.register_tool(Tool(
            name="create_long_term_memory",
            description=(
                "创建长期记忆卡片。用于存储需要长期保存的抽象化知识，如用户偏好、规则、重要事件等。"
                "传入cards_json参数，可以是单个记忆卡片的JSON字符串，也可以是JSON列表字符串（批量创建）。"
                "长期记忆字段：topic(主题), abstract_summary(抽象摘要), importance_score(重要性), "
                "memory_type(类型: preference/rule/event/knowledge/characteristic), confidence_score(置信度)。"
            ),
            function=create_long_term_memory_func,
            parameters={
                "type": "object",
                "properties": {
                    "cards_json": {
                        "type": "string",
                        "description": "包含记忆卡片字段的JSON字符串或JSON列表字符串"
                    }
                },
                "required": ["cards_json"]
            }
        ))

        tool_executor.register_tool(Tool(
            name="delete_medium_term_memories",
            description=(
                "删除中期记忆卡片。传入memory_ids列表（要删除的记忆ID）。"
                "用于清理重要程度低且时间较久、或已被合并/提取为长期记忆的原始记忆。"
                "建议先创建长期记忆或合并后再删除，避免数据丢失。"
            ),
            function=delete_medium_term_memories_func,
            parameters={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要删除的记忆ID列表"
                    }
                },
                "required": ["memory_ids"]
            }
        ))

        logger.debug("重整工具注册完成")
