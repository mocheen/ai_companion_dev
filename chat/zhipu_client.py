"""
智谱API客户端
负责与智谱AI API进行交互
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Iterator

logger = logging.getLogger(__name__)


class ZhipuClient:
    """智谱API客户端类"""

    def __init__(self, api_key: str, model: str = "glm-4-flash",
                 base_url: str = "https://open.bigmodel.cn/api/paas/v4/"):
        """
        初始化智谱API客户端

        Args:
            api_key: API密钥
            model: 模型名称
            base_url: API基础URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/') + '/'
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 2000,
             timeout: int = 30,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None,
             stream: bool = False) -> Optional[str]:
        """
        发送聊天请求

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            timeout: 超时时间(秒)
            tools: 工具列表，用于Function Calling（可选）
            tool_choice: 工具选择策略，可选值: "auto", "any", "none", 或指定具体工具（可选）
            stream: 是否使用流式输出

        Returns:
            模型返回的文本内容或Function Calling响应，失败返回None
        """
        if stream:
            full_response = ""
            for chunk in self.chat_stream(messages, temperature, max_tokens, timeout, tools, tool_choice):
                full_response += chunk
            return full_response
        else:
            return self._chat_non_stream(messages, temperature, max_tokens, timeout, tools, tool_choice)

    def _chat_non_stream(self, messages: List[Dict[str, str]],
                        temperature: float = 0.7,
                        max_tokens: int = 2000,
                        timeout: int = 30,
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[str] = None) -> Optional[str]:
        """
        非流式聊天请求（用于Function Calling等场景）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 超时时间(秒)
            tools: 工具列表
            tool_choice: 工具选择策略

        Returns:
            模型返回的文本内容或Function Calling响应，失败返回None
        """
        url = f"{self.base_url}chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
                logger.debug(f"设置tool_choice={tool_choice}以启用函数调用模式")

        try:
            logger.debug(f"发送请求到智谱API: {url}")

            if "tools" in payload:
                logger.debug(f"请求payload包含tools参数，工具数量: {len(payload['tools'])}")
                logger.debug(f"tool_choice参数: {payload.get('tool_choice', '未设置')}")
                if payload.get("tools"):
                    logger.debug(f"第一个tool的格式: {payload['tools'][0]}")

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()
            result = response.json()

            if "tools" in payload:
                logger.debug(f"API完整响应: {json.dumps(result, ensure_ascii=False)[:500]}...")

            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]

                if "tools" in payload:
                    logger.debug(f"响应message的keys: {list(message.keys())}")
                    if "tool_calls" in message:
                        logger.debug(f"响应包含tool_calls，数量: {len(message['tool_calls']) if message['tool_calls'] else 0}")
                    else:
                        logger.debug(f"响应不包含tool_calls字段")

                if "tool_calls" in message and message["tool_calls"]:
                    logger.debug(f"返回Function Calling响应")
                    return message
                elif "content" in message:
                    logger.debug(f"返回文本响应")
                    return message["content"]
                else:
                    logger.warning(f"API响应消息缺少content和tool_calls: {message}")
                    return None
            else:
                logger.error(f"API响应格式异常: {result}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"请求超时 (>{timeout}秒)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = 0.7,
                    max_tokens: int = 2000,
                    timeout: int = 30,
                    tools: Optional[List[Dict[str, Any]]] = None,
                    tool_choice: Optional[str] = None) -> Iterator[str]:
        """
        流式聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 超时时间(秒)
            tools: 工具列表
            tool_choice: 工具选择策略

        Yields:
            流式返回的文本片段
        """
        url = f"{self.base_url}chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        try:
            logger.debug(f"发送流式请求到智谱API: {url}")

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
                stream=True
            )

            response.raise_for_status()

            logger.debug(f"开始解析流式响应...")

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                logger.debug(f"收到流式数据行: {line[:100]}")

                if line.startswith('data: '):
                    data = line[6:]

                    if data == '[DONE]':
                        logger.debug("收到 [DONE] 标记，流式响应结束")
                        break

                    try:
                        chunk = json.loads(data)
                        logger.debug(f"解析后的chunk: {chunk}")
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                logger.debug(f"提取到content: {content}")
                                yield content
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析流式响应失败: {e}, data: {data}")
                        continue

            logger.debug("流式响应解析完成")

        except requests.exceptions.Timeout:
            logger.error(f"请求超时 (>{timeout}秒)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise

    def test_connection(self) -> bool:
        """
        测试API连接

        Returns:
            连接成功返回True，否则返回False
        """
        test_messages = [{"role": "user", "content": "你好"}]
        response = self.chat(test_messages, timeout=10)
        return response is not None

    def get_embeddings(self, texts: List[str], model: str = "embedding-2",
                      timeout: int = 30) -> Optional[List[List[float]]]:
        """
        获取文本的嵌入向量

        Args:
            texts: 文本列表
            model: 嵌入模型名称
            timeout: 超时时间(秒)

        Returns:
            嵌入向量列表，失败返回None
        """
        url = f"{self.base_url}embeddings"

        try:
            results = []
            # 智��API的embedding接口支持批量处理，一次最多处理100个文本
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
                    timeout=timeout
                )

                response.raise_for_status()
                result = response.json()

                if "data" in result and len(result["data"]) > 0:
                    for item in result["data"]:
                        embedding = item["embedding"]
                        results.append(embedding)
                else:
                    logger.error(f"Embedding API响应格式异常: {result}")
                    return None

            return results

        except requests.exceptions.Timeout:
            logger.error(f"请求超时 (>{timeout}秒)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise
