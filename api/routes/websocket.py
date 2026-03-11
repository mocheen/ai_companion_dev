"""
WebSocket路由
提供流式对话和实时日志推送
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Set
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..app import get_chat_manager, get_memory_system, get_command_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.chat_connections: Set[WebSocket] = set()
        self.log_connections: Set[WebSocket] = set()
        self.log_buffer: deque = deque(maxlen=500)  # 日志缓冲区

    async def connect_chat(self, websocket: WebSocket):
        """连接聊天WebSocket"""
        await websocket.accept()
        self.chat_connections.add(websocket)
        logger.info(f"聊天WebSocket连接建立，当前连接数: {len(self.chat_connections)}")

    def disconnect_chat(self, websocket: WebSocket):
        """断开聊天WebSocket"""
        self.chat_connections.discard(websocket)
        logger.info(f"聊天WebSocket断开，当前连接数: {len(self.chat_connections)}")

    async def connect_log(self, websocket: WebSocket):
        """连接日志WebSocket"""
        await websocket.accept()
        self.log_connections.add(websocket)
        logger.info(f"日志WebSocket连接建立，当前连接数: {len(self.log_connections)}")

        # 发送历史日志
        for log_entry in self.log_buffer:
            try:
                await websocket.send_json(log_entry)
            except Exception:
                break

    def disconnect_log(self, websocket: WebSocket):
        """断开日志WebSocket"""
        self.log_connections.discard(websocket)
        logger.info(f"日志WebSocket断开，当前连接数: {len(self.log_connections)}")

    async def broadcast_log(self, log_entry: dict):
        """广播日志消息"""
        self.log_buffer.append(log_entry)
        disconnected = set()

        for connection in self.log_connections:
            try:
                await connection.send_json(log_entry)
            except Exception:
                disconnected.add(connection)

        for conn in disconnected:
            self.log_connections.discard(conn)


# 全局连接管理器
manager = ConnectionManager()


class WebSocketLogHandler(logging.Handler):
    """WebSocket日志处理器"""

    def emit(self, record):
        try:
            log_entry = {
                "type": "log",
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": self.format(record),
                "logger": record.name
            }
            # 使用asyncio安全地广播日志
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(manager.broadcast_log(log_entry))
            except RuntimeError:
                pass
        except Exception:
            self.handleError(record)


# 注册日志处理器
def setup_websocket_logging():
    """设置WebSocket日志处理器"""
    ws_handler = WebSocketLogHandler()
    ws_handler.setLevel(logging.INFO)
    ws_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(ws_handler)


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """
    流式对话WebSocket

    客户端发送格式: {"type": "message", "content": "用户消息"}
    服务端返回格式:
        - {"type": "chunk", "content": "文本片段"}
        - {"type": "done", "message_id": "xxx"}
        - {"type": "error", "message": "错误信息"}
    """
    await manager.connect_chat(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "无效的JSON格式"
                })
                continue

            if message.get("type") == "message":
                user_content = message.get("content", "").strip()

                if not user_content:
                    await websocket.send_json({
                        "type": "error",
                        "message": "消息不能为空"
                    })
                    continue

                # 检查是否为命令
                command_manager = get_command_manager()
                if command_manager:
                    command_result = command_manager.execute(user_content)
                    if command_result is not None:
                        await websocket.send_json({
                            "type": "command_result",
                            "success": command_result.success,
                            "message": command_result.message
                        })
                        continue

                # 流式对话
                chat_manager = get_chat_manager()
                if not chat_manager:
                    await websocket.send_json({
                        "type": "error",
                        "message": "服务未就绪"
                    })
                    continue

                try:
                    full_response = ""
                    async_generator = chat_manager.chat_stream_async(user_content)

                    async for chunk in async_generator:
                        full_response += chunk
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk
                        })

                    await websocket.send_json({
                        "type": "done",
                        "message_id": f"msg_{datetime.now().timestamp()}"
                    })

                except Exception as e:
                    logger.error(f"流式对话失败: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"对话失败: {str(e)}"
                    })

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect_chat(websocket)
    except Exception as e:
        logger.error(f"聊天WebSocket错误: {e}", exc_info=True)
        manager.disconnect_chat(websocket)


@router.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    """
    实时日志WebSocket

    服务端推送格式:
        - {"type": "log", "timestamp": "...", "level": "INFO", "message": "..."}
    """
    await manager.connect_log(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect_log(websocket)
    except Exception as e:
        logger.error(f"日志WebSocket错误: {e}", exc_info=True)
        manager.disconnect_log(websocket)
