"""
LLM API抽象层
支持多种LLM提供商（智谱、OpenAI等）
"""

from .base_llm import BaseLLM
from .zhipu_llm import ZhipuLLM

__all__ = ['BaseLLM', 'ZhipuLLM']
