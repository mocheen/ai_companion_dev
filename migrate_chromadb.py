#!/usr/bin/env python3
"""
ChromaDB 数据迁移脚本
将本地ChromaDB数据迁移到远程ChromaDB服务
"""

import sys
import yaml
import json
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_local_client(persistence_dir: str):
    """获取本地ChromaDB客户端"""
    logger.info(f"连接本地ChromaDB: {persistence_dir}")
    return chromadb.PersistentClient(
        path=persistence_dir,
        settings=Settings(anonymized_telemetry=False)
    )


def get_remote_client(host: str, port: int):
    """获取远程ChromaDB客户端"""
    logger.info(f"连接远程ChromaDB: {host}:{port}")
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False)
    )


def migrate_collection(
    source_client,
    target_client,
    collection_name: str
):
    """迁移单个集合"""
    logger.info(f"开始迁移集合: {collection_name}")
    
    try:
        source_collection = source_client.get_collection(collection_name)
    except Exception as e:
        logger.warning(f"源集合不存在: {collection_name}, 跳过")
        return
    
    data = source_collection.get()
    
    if not data["ids"]:
        logger.info(f"集合 {collection_name} 为空，跳过")
        return
    
    logger.info(f"集合 {collection_name} 包含 {len(data['ids'])} 条记录")
    
    try:
        target_collection = target_client.get_collection(collection_name)
        logger.info(f"目标集合已存在，清空现有数据")
        target_collection.delete(ids=data["ids"])
    except Exception:
        logger.info(f"目标集合不存在，创建新集合")
        target_collection = target_client.create_collection(
            name=collection_name,
            metadata=source_collection.metadata
        )
    
    logger.info(f"正在写入 {len(data['ids'])} 条记录到目标集合...")
    
    target_collection.add(
        ids=data["ids"],
        embeddings=data["embeddings"],
        documents=data["documents"],
        metadatas=data["metadatas"]
    )
    
    logger.info(f"✅ 集合 {collection_name} 迁移完成")


def main():
    """主函数"""
    config = load_config()
    vector_db_config = config.get("vector_db", {})
    
    print("=" * 60)
    print("ChromaDB 数据迁移工具")
    print("=" * 60)
    print()
    
    print("请确认以下信息：")
    print(f"1. 本地ChromaDB目录: {vector_db_config.get('persistence_dir', 'data/chromadb')}")
    print(f"2. 远程ChromaDB地址: {vector_db_config.get('remote_host', 'localhost')}:{vector_db_config.get('remote_port', 8000)}")
    print(f"3. 要迁移的集合: {vector_db_config.get('collection_medium', 'medium_term_memories')}, {vector_db_config.get('collection_long', 'long_term_memories')}")
    print()
    
    confirm = input("确认开始迁移？(yes/no): ").strip().lower()
    if confirm != "yes":
        print("已取消迁移")
        return
    
    try:
        local_client = get_local_client(vector_db_config.get("persistence_dir", "data/chromadb"))
        remote_client = get_remote_client(
            vector_db_config.get("remote_host", "localhost"),
            vector_db_config.get("remote_port", 8000)
        )
        
        print()
        print("-" * 60)
        
        migrate_collection(
            local_client,
            remote_client,
            vector_db_config.get("collection_medium", "medium_term_memories")
        )
        
        migrate_collection(
            local_client,
            remote_client,
            vector_db_config.get("collection_long", "long_term_memories")
        )
        
        print("-" * 60)
        print()
        print("✅ 所有数据迁移完成！")
        print()
        print("现在你可以：")
        print("1. 修改 config/config.yaml 中的 vector_db.mode 为 'remote'")
        print("2. 运行 view_memory.py 验证数据")
        
    except Exception as e:
        logger.error(f"迁移失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
