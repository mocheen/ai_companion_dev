"""
智谱API客户端
负责与智谱AI API进行交互
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any

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
             timeout: int = 30) -> Optional[str]:
        """
        发送聊天请求

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            timeout: 超时时间(秒)

        Returns:
            模型返回的文本内容，失败返回None
        """
        url = f"{self.base_url}chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            logger.debug(f"发送请求到智谱API: {url}")
            logger.debug(f"请求消息: {json.dumps(messages, ensure_ascii=False)}")

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()
            result = response.json()

            # 提取返回的内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.debug(f"API返回: {content}")
                return content
            else:
                logger.error(f"API响应格式异常: {result}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"请求超时 (>{timeout}秒)")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return None

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
            # 智谱API的embedding接口支持批量处理，一次最多处理100个文本
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                payload = {
                    "model": model,
                    "input": batch_texts
                }

                logger.debug(f"发送embedding请求到智谱API: {url}, 批次大小: {len(batch_texts)}")
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
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return None
