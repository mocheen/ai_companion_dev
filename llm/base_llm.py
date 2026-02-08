"""
LLM提供商的抽象基类
定义统一的接口，支持多种LLM服务
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from llm.models import LLMRequest, LLMResponse, Message, Tool


class BaseLLM(ABC):
    """
    LLM提供商的抽象基类

    所有LLM提供商（智谱、OpenAI等）都应该实现这个接口
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            model: 模型名称
            **kwargs: 其他提供商特定的参数
        """
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        """
        发送聊天请求

        Args:
            request: LLM请求对象

        Returns:
            LLM响应对象
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        获取文本的嵌入向量

        Args:
            texts: 文本列表
            model: 嵌入模型名称（可选，使用默认值）

        Returns:
            嵌入向量列表
        """
        pass

    def test_connection(self) -> bool:
        """
        测试API连接

        Returns:
            连接成功返回True，否则返回False
        """
        try:
            test_request = LLMRequest(
                messages=[Message(role="user", content="你好")]
            )
            response = self.chat(test_request)
            return response.content is not None
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False
