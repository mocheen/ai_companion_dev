"""
记忆相关API路由
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..app import get_memory_system

logger = logging.getLogger(__name__)
router = APIRouter()


class MemoryItem(BaseModel):
    """记忆项"""
    id: str
    topic: str
    key_points: List[str]
    tags: List[str]
    importance: float
    emotion: str
    dialogue_type: str
    created_at: str


class MemoryListResponse(BaseModel):
    """记忆列表响应"""
    success: bool
    memories: List[dict]
    count: int


class MemoryDetailResponse(BaseModel):
    """记忆详情响应"""
    success: bool
    memory: Optional[dict] = None


@router.get("/memory/short", response_model=MemoryListResponse)
async def get_short_term_memory():
    """获取短期记忆"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        messages = memory_system.get_short_term_memory()

        memories = [
            {
                "id": getattr(msg, 'message_id', f"short_{i}"),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else ""
            }
            for i, msg in enumerate(messages)
        ]

        return MemoryListResponse(
            success=True,
            memories=memories,
            count=len(memories)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取短期记忆失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/medium", response_model=MemoryListResponse)
async def get_medium_term_memory(
    days: Optional[int] = Query(None, description="最近N天"),
    min_importance: Optional[float] = Query(None, description="最小重要性"),
    max_importance: Optional[float] = Query(None, description="最大重要性"),
    emotion: Optional[str] = Query(None, description="情感类型"),
    limit: Optional[int] = Query(100, description="返回数量限制")
):
    """获取中期记忆"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'vector_store'):
            return MemoryListResponse(
                success=True,
                memories=[],
                count=0
            )

        memories = memory_system.vector_store.get_all_medium_term_memories()

        # 应用过滤
        filtered = []
        for mem in memories:
            data = mem["data"]

            # 时间过滤
            if days is not None:
                created_at = datetime.fromisoformat(data["created_at"])
                cutoff = datetime.now() - timedelta(days=days)
                if created_at < cutoff:
                    continue

            # 重要性过滤
            importance = data.get("importance_score", 0.5)
            if min_importance is not None and importance < min_importance:
                continue
            if max_importance is not None and importance > max_importance:
                continue

            # 情感过滤
            if emotion is not None and data.get("emotion") != emotion:
                continue

            filtered.append(mem)

        # 应用数量限制
        if limit:
            filtered = filtered[:limit]

        return MemoryListResponse(
            success=True,
            memories=filtered,
            count=len(filtered)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取中期记忆失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/long", response_model=MemoryListResponse)
async def get_long_term_memory(
    days: Optional[int] = Query(None, description="最近N天"),
    min_importance: Optional[float] = Query(None, description="最小重要性"),
    memory_type: Optional[str] = Query(None, description="记忆类型"),
    limit: Optional[int] = Query(100, description="返回数量限制")
):
    """获取长期记忆"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'vector_store'):
            return MemoryListResponse(
                success=True,
                memories=[],
                count=0
            )

        memories = memory_system.vector_store.get_all_long_term_memories()

        # 应用过滤
        filtered = []
        for mem in memories:
            data = mem["data"]

            # 时间过滤
            if days is not None:
                created_at = datetime.fromisoformat(data["created_at"])
                cutoff = datetime.now() - timedelta(days=days)
                if created_at < cutoff:
                    continue

            # 重要性过滤
            importance = data.get("importance_score", 0.5)
            if min_importance is not None and importance < min_importance:
                continue

            # 类型过滤
            if memory_type is not None and data.get("memory_type") != memory_type:
                continue

            filtered.append(mem)

        # 应用数量限制
        if limit:
            filtered = filtered[:limit]

        return MemoryListResponse(
            success=True,
            memories=filtered,
            count=len(filtered)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取长期记忆失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/medium/{memory_id}")
async def delete_medium_term_memory(memory_id: str):
    """删除中期记忆"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'vector_store'):
            raise HTTPException(status_code=400, detail="当前记忆系统不支持此操作")

        success = memory_system.vector_store.delete_medium_term_memory(memory_id)

        if success:
            return {"success": True, "message": "记忆已删除"}
        else:
            raise HTTPException(status_code=404, detail="记忆不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除中期记忆失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/long/{memory_id}")
async def delete_long_term_memory(memory_id: str):
    """删除长期记忆"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'vector_store'):
            raise HTTPException(status_code=400, detail="当前记忆系统不支持此操作")

        success = memory_system.vector_store.delete_long_term_memory(memory_id)

        if success:
            return {"success": True, "message": "记忆已删除"}
        else:
            raise HTTPException(status_code=404, detail="记忆不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除长期记忆失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
