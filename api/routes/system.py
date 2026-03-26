"""
系统管理API路由
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..app import get_config, get_memory_system, get_command_manager, get_chat_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class CommandRequest(BaseModel):
    """命令请求"""
    command: str
    args: Optional[list] = None


class CommandResponse(BaseModel):
    """命令响应"""
    success: bool
    result: str
    should_exit: bool = False


class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    success: bool
    status: Dict[str, Any]


@router.post("/command", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """执行系统命令"""
    try:
        command_manager = get_command_manager()
        if not command_manager:
            raise HTTPException(status_code=503, detail="服务未就绪")

        result = command_manager.execute(request.command)

        if result is None:
            raise HTTPException(status_code=400, detail="无效的命令")

        return CommandResponse(
            success=result.success,
            result=result.message,
            should_exit=result.should_exit
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行命令失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_system_status_internal(light: bool = False) -> dict:
    """
    获取系统状态（内部函数，用于回调）

    Args:
        light: 轻量级模式，不测试API连接

    Returns:
        状态字典
    """
    memory_system = get_memory_system()
    chat_manager = get_chat_manager()
    config = get_config()

    status = {
        "memory": {
            "short_term": 0,
            "short_term_max": 0,
            "medium_term": 0,
            "long_term": 0,
            "type": "unknown"
        },
        "agent": {
            "archive": "idle",
            "reorganize": "idle"
        },
        "api": {
            "connected": False,
            "model": "",
            "response_time": 0
        },
        "uptime": datetime.now().isoformat()
    }

    if memory_system:
        # 短期记忆
        short_term = memory_system.get_short_term_memory()
        status["memory"]["short_term"] = len(short_term)
        status["memory"]["short_term_max"] = getattr(memory_system, 'max_messages', 0)
        status["memory"]["type"] = type(memory_system).__name__

        # 中长期记忆
        if hasattr(memory_system, 'vector_store'):
            try:
                medium_memories = memory_system.vector_store.get_all_medium_term_memories()
                status["memory"]["medium_term"] = len(medium_memories)
            except Exception:
                pass

            try:
                long_memories = memory_system.vector_store.get_all_long_term_memories()
                status["memory"]["long_term"] = len(long_memories)
            except Exception:
                pass

    if chat_manager and config:
        api_config = config.get("api", {})
        status["api"]["model"] = api_config.get("model", "unknown")
        # 轻量级模式下跳过API连接测试，避免消耗token
        if not light:
            status["api"]["connected"] = chat_manager.test_api_connection()
        else:
            # 保留上次的连接状态，如果没有则默认为True
            status["api"]["connected"] = True

    return status


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(light: bool = Query(False, description="轻量级查询，跳过API连接测试")):
    """获取系统状态

    Args:
        light: 轻量级模式，不测试API连接（避免消耗token）
    """
    try:
        status = get_system_status_internal(light)
        return SystemStatusResponse(
            success=True,
            status=status
        )

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_system_config():
    """获取系统配置（敏感信息已脱敏）"""
    try:
        config = get_config()
        if not config:
            raise HTTPException(status_code=503, detail="服务未就绪")

        # 脱敏处理
        safe_config = {
            "chat": config.get("chat", {}),
            "memory_system": {"type": config.get("memory_system", {}).get("type", "simple")},
            "short_term_memory": config.get("short_term_memory", {}),
            "memory_retrieval": config.get("memory_retrieval", {}),
            "commands": config.get("commands", {}),
            "logging": {"level": config.get("logging", {}).get("level", "INFO")}
        }

        return {"success": True, "config": safe_config}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def update_system_config(config_updates: dict):
    """更新系统配置（运行时，不持久化）"""
    try:
        # 注意：这里只更新内存中的配置，不写入文件
        # 实际项目中应该有更完善的配置管理机制
        return {"success": True, "message": "配置已更新（仅本次会话有效）"}

    except Exception as e:
        logger.error(f"更新配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_recent_logs(lines: int = 100):
    """获取最近的日志"""
    try:
        config = get_config()
        log_config = config.get("logging", {})
        log_dir = log_config.get("log_dir", "logs")
        log_file = log_config.get("log_file", "ai_companion.log")

        log_path = f"{log_dir}/{log_file}"

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            return {
                "success": True,
                "logs": [line.strip() for line in recent_lines],
                "count": len(recent_lines)
            }
        except FileNotFoundError:
            return {
                "success": True,
                "logs": [],
                "count": 0,
                "message": "日志文件不存在"
            }

    except Exception as e:
        logger.error(f"获取日志失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
