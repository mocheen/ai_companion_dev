"""
智谱AI的LLM实现
完全兼容智谱API的工具调用格式
"""

import requests
import json
import logging
from typing import List, Optional, Dict, Any

from llm.base_llm import BaseLLM
from llm.models import (
    LLMRequest, LLMResponse, Message, Tool,
    ToolCall, FunctionCall, MessageRole
)

logger = logging.getLogger(__name__)


class ZhipuLLM(BaseLLM):
    """
    智谱AI的LLM实现

    支持功能：
    - 普通对话
    - 工具调用（Function Calling）
    - 文本嵌入
    """

    def __init__(self, api_key: str, model: str = "glm-4-flash",
                 base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
                 **kwargs):
        """
        初始化智谱LLM客户端

        Args:
            api_key: API密钥
            model: 模型名称（默认glm-4-flash）
            base_url: API基础URL
            **kwargs: 其他参数（目前未使用）
        """
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip('/') + '/'
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, request: LLMRequest) -> LLMResponse:
        """
        发送聊天请求到智谱API

        Args:
            request: LLM请求对象

        Returns:
            LLM响应对象
        """
        url = f"{self.base_url}chat/completions"

        # 构建请求payload
        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        # 添加工具定义（tools已经是字典列表格式）
        if request.tools:
            # ToolExecutor.get_tools_for_llm()返回的是字典列表
            if isinstance(request.tools[0], dict):
                payload["tools"] = request.tools
            else:
                # 如果是Tool对象，转换为字典
                payload["tools"] = [tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in request.tools]

        # 添加工具选择策略
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        try:
            logger.debug(f"发送请求到智谱API: {url}")
            if request.tools:
                logger.debug(f"请求包含工具定义，数量: {len(request.tools)}")
                logger.debug(f"tool_choice: {request.tool_choice}")

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=request.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.debug(f"API响应状态码: {response.status_code}")

            # 解析响应
            return self._parse_response(result)

        except requests.exceptions.Timeout:
            logger.error(f"请求超时 (>{request.timeout}秒)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise

    def _parse_response(self, result: Dict[str, Any]) -> LLMResponse:
        """
        解析智谱API响应

        Args:
            result: API返回的JSON数据

        Returns:
            LLM响应对象
        """
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error(f"API响应格式异常: {result}")
            return LLMResponse(content="")

        choice = result["choices"][0]
        message_data = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "unknown")

        # 提取文本内容
        content = message_data.get("content")

        # 提取工具调用
        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = []
            for tc_data in message_data["tool_calls"]:
                tool_call = ToolCall(
                    id=tc_data["id"],
                    type=tc_data.get("type", "function"),
                    function=FunctionCall(
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"]
                    )
                )
                tool_calls.append(tool_call)

            logger.debug(f"解析到工具调用，数量: {len(tool_calls)}")

        # 提取token使用情况
        usage = result.get("usage")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )

    def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        获取文本的嵌入向量

        Args:
            texts: 文本列表
            model: 嵌入模型名称（默认使用embedding-2）

        Returns:
            嵌入向量列表
        """
        if model is None:
            model = "embedding-2"

        url = f"{self.base_url}embeddings"

        try:
            results = []
            # 智谱API支持批量处理，每次最多100个文本
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                payload = {
                    "model": model,
                    "input": batch_texts
                }

                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

                if "data" in result and len(result["data"]) > 0:
                    for item in result["data"]:
                        embedding = item["embedding"]
                        results.append(embedding)
                else:
                    logger.error(f"Embedding API响应格式异常: {result}")
                    return []

            return results

        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            raise
