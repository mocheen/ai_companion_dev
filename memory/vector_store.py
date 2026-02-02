"""
向量数据库存储层
负责使用ChromaDB存储和检索记忆卡片
"""

import os
import logging
import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings

from memory.memory_models import MediumTermMemory, LongTermMemory
from chat.zhipu_client import ZhipuClient

logger = logging.getLogger(__name__)


class VectorStore:
    """向量数据库存储类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量数据库

        Args:
            config: 配置字典
        """
        self.config = config
        vector_db_config = config.get("vector_db", {})

        # 向量数据库持久化目录
        self.persistence_dir = vector_db_config.get("persistence_dir", "data/chromadb")
        os.makedirs(self.persistence_dir, exist_ok=True)

        # 嵌入方式
        self.embedding_type = vector_db_config.get("embedding_type", "api")

        if self.embedding_type == "api":
            # 使用智谱AI的embedding API
            api_config = config.get("api", {})
            embedding_api_config = vector_db_config.get("embedding_api_key", api_config.get("api_key", ""))
            embedding_base_url = vector_db_config.get("embedding_base_url", api_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/"))
            embedding_model_name = vector_db_config.get("embedding_model", "embedding-2")

            logger.info(f"使用智谱AI Embedding API: {embedding_model_name}")
            self.zhipu_client = ZhipuClient(
                api_key=embedding_api_config,
                model=embedding_model_name,
                base_url=embedding_base_url
            )
            self.embedding_model_name = embedding_model_name
        else:
            # 使用本地模型
            from sentence_transformers import SentenceTransformer
            self.embedding_model_name = vector_db_config.get(
                "local_embedding_model",
                "shibing624/text2vec-base-chinese"
            )
            logger.info(f"加载本地嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.zhipu_client = None

        # 集合名称
        self.collection_medium_name = vector_db_config.get("collection_medium", "medium_term_memories")
        self.collection_long_name = vector_db_config.get("collection_long", "long_term_memories")

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=self.persistence_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self._init_collections()

        logger.info("向量数据库初始化完成")

    def _init_collections(self):
        """初始化记忆集合"""
        # 获取或创建中期记忆集合
        try:
            self.collection_medium = self.client.get_collection(self.collection_medium_name)
            logger.info(f"加载中期记忆集合: {self.collection_medium_name}")
        except Exception:
            self.collection_medium = self.client.create_collection(
                name=self.collection_medium_name,
                metadata={"description": "中期记忆卡片"}
            )
            logger.info(f"创建中期记忆集合: {self.collection_medium_name}")

        # 获取或创建长期记忆集合
        try:
            self.collection_long = self.client.get_collection(self.collection_long_name)
            logger.info(f"加载长期记忆集合: {self.collection_long_name}")
        except Exception:
            self.collection_long = self.client.create_collection(
                name=self.collection_long_name,
                metadata={"description": "长期记忆卡片"}
            )
            logger.info(f"创建长期记忆集合: {self.collection_long_name}")

    def _encode_text(self, text: str) -> List[float]:
        """
        将文本编码为向量

        Args:
            text: 输入文本

        Returns:
            向量表示

        Raises:
            Exception: 当向量编码失败时抛出异常
        """
        if self.embedding_type == "api":
            # 使用智谱AI的embedding API
            embeddings = self.zhipu_client.get_embeddings([text], model=self.embedding_model_name)
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                raise ValueError(f"获取embedding返回空，文本: {text[:50]}...")
        else:
            # 使用本地模型
            return self.embedding_model.encode(text).tolist()

    def add_medium_term_memory(self, memory: MediumTermMemory) -> str:
        """
        添加中期记忆

        Args:
            memory: 中期记忆对象

        Returns:
            记忆ID

        Raises:
            Exception: 当向量编码失败时抛出异常
        """
        memory_id = str(uuid.uuid4())
        text = memory.to_search_text()
        embedding = self._encode_text(text)

        self.collection_medium.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "medium",
                "created_at": memory.created_at.isoformat(),
                "importance_score": memory.importance_score,
                "data": json.dumps(memory.to_dict(), ensure_ascii=False)
            }]
        )

        logger.debug(f"添加中期记忆: {memory_id} - {memory.topic_summary[:50]}...")
        return memory_id

    def add_long_term_memory(self, memory: LongTermMemory) -> str:
        """
        添加长期记忆

        Args:
            memory: 长期记忆对象

        Returns:
            记忆ID

        Raises:
            Exception: 当向量编码失败时抛出异常
        """
        memory_id = str(uuid.uuid4())
        text = memory.to_search_text()
        embedding = self._encode_text(text)

        self.collection_long.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "long",
                "created_at": memory.created_at.isoformat(),
                "importance_score": memory.importance_score,
                "memory_type": memory.memory_type.value,
                "data": json.dumps(memory.to_dict(), ensure_ascii=False)
            }]
        )

        logger.debug(f"添加长期记忆: {memory_id} - {memory.topic[:50]}...")
        return memory_id

    def search_medium_term_memories(
        self,
        query: str,
        n_results: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索中期记忆

        Args:
            query: 查询文本
            n_results: 返回结果数量
            filters: 过滤条件

        Returns:
            匹配的记忆列表

        Raises:
            Exception: 当向量编码或检索失败时抛出异常
        """
        # 检查集合是否为空，为空则直接返回空列表，避免不必要的API调用
        try:
            count = self.collection_medium.count()
            if count == 0:
                logger.debug("中期记忆集合为空，跳过检索")
                return []
        except Exception as e:
            logger.warning(f"检查集合数量失败: {e}")

        query_embedding = self._encode_text(query)

        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if filters:
            kwargs["where"] = filters

        results = self.collection_medium.query(**kwargs)

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                data = json.loads(metadata["data"])

                memories.append({
                    "id": memory_id,
                    "distance": results["distances"][0][i] if "distances" in results else 0,
                    "data": data
                })

        return memories

    def search_long_term_memories(
        self,
        query: str,
        n_results: int = 2,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索长期记忆

        Args:
            query: 查询文本
            n_results: 返回结果数量
            filters: 过滤条件

        Returns:
            匹配的记忆列表

        Raises:
            Exception: 当向量编码或检索失败时抛出异常
        """
        # 检查集合是否为空，为空则直接返回空列表，避免不必要的API调用
        try:
            count = self.collection_long.count()
            if count == 0:
                logger.debug("长期记忆集合为空，跳过检索")
                return []
        except Exception as e:
            logger.warning(f"检查集合数量失败: {e}")

        query_embedding = self._encode_text(query)

        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if filters:
            kwargs["where"] = filters

        results = self.collection_long.query(**kwargs)

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                data = json.loads(metadata["data"])

                memories.append({
                    "id": memory_id,
                    "distance": results["distances"][0][i] if "distances" in results else 0,
                    "data": data
                })

        return memories

    def get_all_medium_term_memories(self) -> List[Dict[str, Any]]:
        """
        获取所有中期记忆

        Returns:
            所有中期记忆列表

        Raises:
            Exception: 当获取记忆失败时抛出异常
        """
        results = self.collection_medium.get()

        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                data = json.loads(metadata["data"])

                memories.append({
                    "id": memory_id,
                    "data": data
                })

        return memories

    def get_all_long_term_memories(self) -> List[Dict[str, Any]]:
        """
        获取所有长期记忆

        Returns:
            所有长期记忆列表

        Raises:
            Exception: 当获取记忆失败时抛出异常
        """
        results = self.collection_long.get()

        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                data = json.loads(metadata["data"])

                memories.append({
                    "id": memory_id,
                    "data": data
                })

        return memories

    def delete_medium_term_memory(self, memory_id: str) -> bool:
        """
        删除中期记忆

        Args:
            memory_id: 记忆ID

        Returns:
            成功返回True
        """
        try:
            self.collection_medium.delete(ids=[memory_id])
            logger.debug(f"删除中期记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除中期记忆失败: {e}")
            return False

    def delete_long_term_memory(self, memory_id: str) -> bool:
        """
        删除长期记忆

        Args:
            memory_id: 记忆ID

        Returns:
            成功返回True
        """
        try:
            self.collection_long.delete(ids=[memory_id])
            logger.debug(f"删除长期记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除长期记忆失败: {e}")
            return False

    def update_medium_term_memory(self, memory_id: str, memory: MediumTermMemory) -> bool:
        """
        更新中期记忆

        Args:
            memory_id: 记忆ID
            memory: 新的记忆对象

        Returns:
            成功返回True

        Raises:
            Exception: 当向量编码或更新失败时抛出异常
        """
        text = memory.to_search_text()
        embedding = self._encode_text(text)

        self.collection_medium.update(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "medium",
                "created_at": memory.created_at.isoformat(),
                "importance_score": memory.importance_score,
                "data": json.dumps(memory.to_dict(), ensure_ascii=False)
            }]
        )

        logger.debug(f"更新中期记忆: {memory_id}")
        return True

    def update_long_term_memory(self, memory_id: str, memory: LongTermMemory) -> bool:
        """
        更新长期记忆

        Args:
            memory_id: 记忆ID
            memory: 新的记忆对象

        Returns:
            成功返回True

        Raises:
            Exception: 当向量编码或更新失败时抛出异常
        """
        text = memory.to_search_text()
        embedding = self._encode_text(text)

        self.collection_long.update(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "long",
                "created_at": memory.created_at.isoformat(),
                "importance_score": memory.importance_score,
                "memory_type": memory.memory_type.value,
                "data": json.dumps(memory.to_dict(), ensure_ascii=False)
            }]
        )

        logger.debug(f"更新长期记忆: {memory_id}")
        return True
