"""
回溯管理API路由
提供回溯快照的查看、列表和回滚执行接口
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..app import get_memory_system

logger = logging.getLogger(__name__)
router = APIRouter()


class RollbackSnapshotSummary(BaseModel):
    """快照摘要"""
    snapshot_id: str
    parent_id: Optional[str] = None
    agent_type: str
    created_at: str
    operations_count: int
    filename: str


class RollbackListResponse(BaseModel):
    """快照列表响应"""
    success: bool
    snapshots: List[dict]
    count: int
    agent_running: bool


class RollbackDetailResponse(BaseModel):
    """快照详情响应"""
    success: bool
    snapshot: Optional[dict] = None


class RollbackExecuteResponse(BaseModel):
    """回滚执行响应"""
    success: bool
    message: str
    rolled_back_count: Optional[int] = None
    operations_rolled_back: Optional[int] = None
    deleted_files: Optional[List[str]] = None


@router.get("/rollback/list", response_model=RollbackListResponse)
async def list_rollback_snapshots():
    """获取所有回溯快照列表"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'rollback_manager'):
            return RollbackListResponse(
                success=True, snapshots=[], count=0, agent_running=False
            )

        rb = memory_system.rollback_manager
        snapshots = rb.get_all_snapshots()
        agent_running = rb.is_agent_running()

        return RollbackListResponse(
            success=True,
            snapshots=snapshots,
            count=len(snapshots),
            agent_running=agent_running,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回溯快照列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback/{snapshot_id}", response_model=RollbackDetailResponse)
async def get_rollback_snapshot(snapshot_id: str):
    """获取指定回溯快照的详细信息"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'rollback_manager'):
            raise HTTPException(status_code=400, detail="回溯功能不可用")

        rb = memory_system.rollback_manager
        detail = rb.get_snapshot_detail(snapshot_id)

        if not detail:
            raise HTTPException(status_code=404, detail="快照不存在")

        return RollbackDetailResponse(success=True, snapshot=detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回溯快照详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/{snapshot_id}/execute", response_model=RollbackExecuteResponse)
async def execute_rollback(snapshot_id: str):
    """执行回溯到指定快照节点"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not hasattr(memory_system, 'rollback_manager'):
            raise HTTPException(status_code=400, detail="回溯功能不可用")

        rb = memory_system.rollback_manager

        if rb.is_agent_running():
            raise HTTPException(
                status_code=409,
                detail="Agent正在运行中（归档或重整），无法执行回溯，请稍后再试"
            )

        result = rb.rollback_to(
            snapshot_id=snapshot_id,
            vector_store=memory_system.vector_store,
            redis_client=memory_system.redis_client,
            redis_key_short_term=memory_system.redis_key_short_term,
            redis_key_flow_state=memory_system.redis_key_flow_state,
        )

        if result["success"]:
            return RollbackExecuteResponse(
                success=True,
                message=result["message"],
                rolled_back_count=result.get("rolled_back_count"),
                operations_rolled_back=result.get("operations_rolled_back"),
                deleted_files=result.get("deleted_files"),
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行回溯失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
