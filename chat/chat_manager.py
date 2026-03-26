"""
对话管理器
负责管理对话流程，协调智谱API和记忆系统
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator
from .zhipu_client import ZhipuClient

logger = logging.getLogger(__name__)


class ChatMessage:
    """聊天消息类"""

    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        初始化聊天消息

        Args:
            role: 消息角色 ("user" 或 "assistant")
            content: 消息内容
            timestamp: 时间戳
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式，用于API调用"""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建实例"""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp
        )

    def __str__(self) -> str:
        """格式化输出"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        role_name = "用户" if self.role == "user" else "AI"
        return f"[{time_str}] {role_name}: {self.content}"


class ChatManager:
    """对话管理器类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话管理器

        Args:
            config: 配置字典
        """
        self.config = config

        # 初始化智谱API客户端
        api_config = config.get("api", {})
        self.zhipu_client = ZhipuClient(
            api_key=api_config.get("api_key", ""),
            model=api_config.get("model", "glm-4-flash"),
            base_url=api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/")
        )

        # 加载提示模板
        self._load_prompts()

        # 记忆系统接口（预留）
        self.memory_system = None

    def _load_prompts(self):
        """加载系统提示和用户提示模板"""
        prompts_config = self.config.get("prompts", {})

        # 加载系统提示
        system_prompt_path = prompts_config.get("system_prompt", "prompts/system_prompt.txt")
        self.system_prompt = self._read_file(system_prompt_path)

        # 加载用户提示模板
        user_prompt_path = prompts_config.get("user_prompt", "prompts/user_prompt.txt")
        self.user_prompt_template = self._read_file(user_prompt_path)

    def _read_file(self, file_path: str) -> str:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            文件内容，失败返回空字符串
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return ""

    def set_memory_system(self, memory_system):
        """
        设置记忆系统

        Args:
            memory_system: 记忆系统实例
        """
        self.memory_system = memory_system
        logger.info("记忆系统已设置")

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """
        格式化消息列表为文本

        Args:
            messages: 消息列表

        Returns:
            格式化后的文本
        """
        if not messages:
            return "（无）"

        return "\n".join(str(msg) for msg in messages)

    def _build_user_prompt(self, user_message: str,
                           short_term_memory: List[ChatMessage],
                           long_term_memory: str = "",
                           medium_term_memory: str = "") -> str:
        """
        构建用户提示

        Args:
            user_message: 用户消息
            short_term_memory: 短期记忆（最近对话）
            long_term_memory: 长期记忆
            medium_term_memory: 中期记忆

        Returns:
            构建好的用户提示
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return self.user_prompt_template.format(
            current_time=current_time,
            short_term_memory=self._format_messages(short_term_memory),
            long_term_memory=long_term_memory or "（无）",
            medium_term_memory=medium_term_memory or "（无）",
            user_message=user_message
        )

    def chat(self, user_message: str) -> Optional[str]:
        """
        与AI对话

        Args:
            user_message: 用户消息

        Returns:
            AI回复内容，失败返回None
        """
        logger.info(f"用户消息: {user_message}")

        # 从记忆系统获取数据
        short_term_memory = []
        long_term_memory = ""
        medium_term_memory = ""

        if self.memory_system:
            # 获取短期记忆
            short_term_memory = self.memory_system.get_short_term_memory()

            # 获取长期记忆
            long_term_memory = self.memory_system.get_long_term_memory(user_message)

            # 获取中期记忆
            medium_term_memory = self.memory_system.get_medium_term_memory(user_message)

        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加用户提示（包含记忆信息）
        user_prompt = self._build_user_prompt(
            user_message,
            short_term_memory,
            long_term_memory,
            medium_term_memory
        )
        messages.append({"role": "user", "content": user_prompt})

        # 记录debug级别的日志
        logger.debug(f"系统提示: {self.system_prompt}")
        logger.debug(f"用户提示: {user_prompt}")

        # 调用智谱API
        chat_config = self.config.get("chat", {})
        response = self.zhipu_client.chat(
            messages=messages,
            temperature=chat_config.get("temperature", 0.7),
            max_tokens=chat_config.get("max_tokens", 2000),
            timeout=chat_config.get("timeout", 30),
            stream=True
        )

        if response is None:
            logger.error("AI回复失败")
            return None

        # 只在INFO级别显示摘要（前100字符）
        response_preview = response[:100] + "..." if len(response) > 100 else response
        logger.info(f"AI回复: {response_preview}")

        # 将对话记录提供给记忆系统
        if self.memory_system:
            self.memory_system.add_conversation(user_message, response)

        return response

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """
        与AI对话（流式输出）

        Args:
            user_message: 用户消息

        Yields:
            流式返回的文本片段
        """
        logger.info(f"用户消息: {user_message}")

        # 从记忆系统获取数据
        short_term_memory = []
        long_term_memory = ""
        medium_term_memory = ""

        if self.memory_system:
            # 获取短期记忆
            short_term_memory = self.memory_system.get_short_term_memory()

            # 获取长期记忆
            long_term_memory = self.memory_system.get_long_term_memory(user_message)

            # 获取中期记忆
            medium_term_memory = self.memory_system.get_medium_term_memory(user_message)

        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加用户提示（包含记忆信息）
        user_prompt = self._build_user_prompt(
            user_message,
            short_term_memory,
            long_term_memory,
            medium_term_memory
        )
        messages.append({"role": "user", "content": user_prompt})

        # 记录debug级别的日志
        logger.debug(f"系统提示: {self.system_prompt}")
        logger.debug(f"用户提示: {user_prompt}")

        # 调用智谱API（流式）
        chat_config = self.config.get("chat", {})
        full_response = ""

        try:
            for chunk in self.zhipu_client.chat_stream(
                messages=messages,
                temperature=chat_config.get("temperature", 0.7),
                max_tokens=chat_config.get("max_tokens", 2000),
                timeout=chat_config.get("timeout", 30)
            ):
                full_response += chunk
                yield chunk

            # 只在INFO级别显示摘要（前100字符）
            response_preview = full_response[:100] + "..." if len(full_response) > 100 else full_response
            logger.info(f"AI回复: {response_preview}")

            # 将对话记录提供给记忆系统
            if self.memory_system:
                self.memory_system.add_conversation(user_message, full_response)

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            raise

    def test_api_connection(self) -> bool:
        """
        测试API连接

        Returns:
            连接成功返回True
        """
        logger.info("测试API连接...")
        return self.zhipu_client.test_connection()

    async def chat_stream_async(self, user_message: str) -> AsyncIterator[str]:
        """
        与AI对话（异步流式输出）

        Args:
            user_message: 用户消息

        Yields:
            流式返回的文本片段
        """
        logger.info(f"用户消息: {user_message}")

        # 从记忆系统获取数据
        short_term_memory = []
        long_term_memory = ""
        medium_term_memory = ""

        if self.memory_system:
            # 获取短期记忆
            short_term_memory = self.memory_system.get_short_term_memory()

            # 获取长期记忆
            long_term_memory = self.memory_system.get_long_term_memory(user_message)

            # 获取中期记忆
            medium_term_memory = self.memory_system.get_medium_term_memory(user_message)

        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加用户提示（包含记忆信息）
        user_prompt = self._build_user_prompt(
            user_message,
            short_term_memory,
            long_term_memory,
            medium_term_memory
        )
        messages.append({"role": "user", "content": user_prompt})

        # 记录debug级别的日志
        logger.debug(f"系统提示: {self.system_prompt}")
        logger.debug(f"用户提示: {user_prompt}")

        # 调用智谱API（流式）
        chat_config = self.config.get("chat", {})
        full_response = ""

        try:
            # 在线程池中运行同步的流式请求
            loop = asyncio.get_event_loop()

            def sync_stream():
                logger.debug("开始执行同步流式请求...")
                result = list(self.zhipu_client.chat_stream(
                    messages=messages,
                    temperature=chat_config.get("temperature", 0.7),
                    max_tokens=chat_config.get("max_tokens", 2000),
                    timeout=chat_config.get("timeout", 30)
                ))
                logger.debug(f"同步流式请求完成，获得 {len(result)} 个chunks")
                return result

            chunks = await loop.run_in_executor(None, sync_stream)
            logger.debug(f"开始发送 {len(chunks)} 个chunks到WebSocket")

            for chunk in chunks:
                full_response += chunk
                yield chunk

            logger.debug(f"所有chunks发送完成，总长度: {len(full_response)}")

            # 只在INFO级别显示摘要（前100字符）
            response_preview = full_response[:100] + "..." if len(full_response) > 100 else full_response
            logger.info(f"AI回复: {response_preview}")

            # 将对话记录提供给记忆系统
            if self.memory_system:
                self.memory_system.add_conversation(user_message, full_response)

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            raise
