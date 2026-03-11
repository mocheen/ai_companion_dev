"""
对话相关API路由
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..app import get_chat_manager, get_memory_system

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """对话请求"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """对话响应"""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: str
    timestamp: str
    message_id: Optional[str] = None


class HistoryResponse(BaseModel):
    """历史记录响应"""
    success: bool
    messages: List[MessageItem]
    count: int


@router.post("/chat", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    发送消息

    注意：此接口不返回AI回复内容，仅确认消息已接收。
    对于流式对话，请使用WebSocket接口 /ws/chat
    """
    try:
        chat_manager = get_chat_manager()
        if not chat_manager:
            raise HTTPException(status_code=503, detail="服务未就绪")

        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        logger.info(f"收到消息: {request.message[:50]}...")

        return ChatResponse(
            success=True,
            message_id=f"msg_{datetime.now().timestamp()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"发送消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
async def get_history():
    """获取对话历史（短期记忆）"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        messages = memory_system.get_short_term_memory()

        message_items = [
            MessageItem(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else "",
                message_id=getattr(msg, 'message_id', None)
            )
            for msg in messages
        ]

        return HistoryResponse(
            success=True,
            messages=message_items,
            count=len(message_items)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取历史记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_history():
    """清空对话历史"""
    try:
        memory_system = get_memory_system()
        if not memory_system:
            raise HTTPException(status_code=503, detail="服务未就绪")

        # 清空短期记忆
        if hasattr(memory_system, 'short_term_memory'):
            memory_system.short_term_memory.clear()
            if hasattr(memory_system, '_save_short_term_memory'):
                memory_system._save_short_term_memory()

        return {"success": True, "message": "历史记录已清空"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空历史记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
