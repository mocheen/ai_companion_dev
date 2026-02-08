"""
自主Agent基类
实现完整的工具调用循环
"""

import logging
from typing import List, Optional
from llm.models import Message, LLMRequest, LLMResponse, MessageRole
from agent.tool import ToolExecutor
from llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    自主Agent基类

    实现完整的工具调用循环：
    1. 发送消息和工具定义给LLM
    2. 检查LLM是否调用工具
    3. 如果调用工具，执行工具并将结果返回给LLM
    4. 重复步骤2-3，直到LLM不再调用工具
    5. 返回最终结果
    """

    def __init__(self, llm: BaseLLM, tool_executor: ToolExecutor,
                 system_prompt: str = "", max_iterations: int = 10):
        """
        初始化Agent

        Args:
            llm: LLM客户端
            tool_executor: 工具执行器
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm = llm
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # 对话历史
        self.messages: List[Message] = []

        # 如果提供了系统提示，添加到消息历史
        if system_prompt:
            self.messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

    def run(self, user_input: str, tool_choice: str = "auto") -> str:
        """
        运行Agent

        Args:
            user_input: 用户输入
            tool_choice: 工具选择策略（auto、any、none等）

        Returns:
            Agent的最终回复
        """
        # 添加用户消息
        self.messages.append(Message(role=MessageRole.USER, content=user_input))

        # 开始循环
        for iteration in range(self.max_iterations):
            logger.debug(f"Agent迭代 {iteration + 1}/{self.max_iterations}")

            # 构建请求
            request = LLMRequest(
                messages=self.messages.copy(),
                tools=self.tool_executor.get_tools_for_llm(),
                tool_choice=tool_choice
            )

            # 调用LLM
            response = self.llm.chat(request)

            # 添加助手响应到历史
            assistant_message = response.to_message()
            self.messages.append(assistant_message)

            # 检查是否有工具调用
            if response.has_tool_calls():
                logger.debug(f"LLM调用了 {len(response.tool_calls)} 个工具")

                # 执行所有工具调用
                for tool_call in response.tool_calls:
                    result = self._execute_tool_call(tool_call)

                    # 添加工具返回消息到历史
                    tool_message = Message(
                        role=MessageRole.TOOL,
                        content=result,
                        tool_call_id=tool_call.id
                    )
                    self.messages.append(tool_message)

                # 继续循环，让LLM根据工具结果继续决策
                continue
            else:
                # 没有工具调用，任务完成
                logger.debug("Agent完成，返回最终结果")
                return response.content or ""

        # 达到最大迭代次数
        logger.warning(f"Agent达到最大迭代次数({self.max_iterations})，强制结束")
        # 返回最后一次助手消息的内容
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT and msg.content:
                return msg.content
        return "Agent执行失败：达到最大迭代次数"

    def _execute_tool_call(self, tool_call) -> str:
        """
        执行单个工具调用

        Args:
            tool_call: 工具调用对象

        Returns:
            工具执行结果
        """
        function = tool_call.function
        logger.debug(f"执行工具: {function.name}, 参数: {function.arguments}")

        result = self.tool_executor.execute_tool_call(
            tool_name=function.name,
            arguments=function.arguments
        )

        logger.debug(f"工具执行结果: {result[:200]}...")
        return result

    def reset(self):
        """重置Agent状态（清空对话历史，保留系统提示）"""
        self.messages = []
        if self.system_prompt:
            self.messages.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))
        logger.debug("Agent状态已重置")

    def get_conversation_history(self) -> List[Message]:
        """
        获取对话历史

        Returns:
            消息列表
        """
        return self.messages.copy()
