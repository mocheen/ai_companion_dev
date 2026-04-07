"""
FastAPI应用工厂
创建和配置FastAPI应用
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from utils import load_config, setup_logging
from chat import ChatManager
from memory import SimpleMemorySystem, FullMemorySystem
from command_manager import CommandManager
from config import ConfigManager

logger = logging.getLogger(__name__)

# 全局实例
_config = None
_chat_manager: Optional[ChatManager] = None
_memory_system = None
_command_manager: Optional[CommandManager] = None
_status_manager = None
_config_manager: Optional[ConfigManager] = None


def get_config():
    """获取配置"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _set_config(new_config):
    """设置全局配置（由配置更新流程调用）"""
    global _config
    _config = new_config


def get_chat_manager() -> ChatManager:
    """获取对话管理器"""
    global _chat_manager
    return _chat_manager


def get_memory_system():
    """获取记忆系统"""
    global _memory_system
    return _memory_system


def get_command_manager() -> CommandManager:
    """获取命令管理器"""
    global _command_manager
    return _command_manager


def get_status_manager():
    """获取状态管理器"""
    global _status_manager
    return _status_manager


def get_config_manager() -> ConfigManager:
    """获取配置管理器"""
    global _config_manager
    return _config_manager


def _rebuild_components(config: dict, full_rebuild: bool = False):
    """根据新配置重建受影响的组件

    Args:
        config: 新的配置字典
        full_rebuild: 是否需要完整重建（API Key/模型等变更时）
    """
    global _chat_manager, _memory_system

    if full_rebuild:
        # 重建 ChatManager（ZhipuClient 内部缓存了 api_key/model/base_url）
        logger.info("完整重建组件...")
        old_memory = _memory_system
        _chat_manager = ChatManager(config)

        # 重建记忆系统
        memory_config = config.get("memory_system", {})
        memory_type = memory_config.get("type", "simple")

        if memory_type == "full":
            logger.info("重建完整记忆系统")
            _memory_system = FullMemorySystem(config)
        else:
            logger.info("重建简单记忆系统")
            short_term_config = config.get("short_term_memory", {})
            max_turns = short_term_config.get("max_turns", 12)
            persist_file = short_term_config.get("persist_file", "data/short_term_memory.json")
            _memory_system = SimpleMemorySystem(max_turns=max_turns, persist_file=persist_file)

        _chat_manager.set_memory_system(_memory_system)

        # 重新设置状态更新回调
        if _status_manager:
            def status_update_callback(status: dict = None):
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        if status is None:
                            from .routes.system import get_system_status_internal
                            status = get_system_status_internal(light=True)
                        asyncio.run_coroutine_threadsafe(
                            _status_manager.broadcast_status(status),
                            loop
                        )
                except Exception as e:
                    logger.error(f"广播状态更新失败: {e}")
            _memory_system.set_status_update_callback(status_update_callback)

        # 重新注册命令
        if _command_manager:
            _command_manager.register_default_commands(_memory_system)

        logger.info("组件重建完成")
    else:
        # 只更新 config 引用（ChatManager 和 FullMemorySystem 已经是动态读取 config 的）
        if _chat_manager:
            _chat_manager.config = config
        if _memory_system and hasattr(_memory_system, 'config'):
            _memory_system.config = config
        logger.info("配置引用已更新（无需重建组件）")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _chat_manager, _memory_system, _command_manager, _config_manager

    # 启动时初始化
    config = get_config()

    # 初始化配置管理器
    _config_manager = ConfigManager()

    # 配置日志
    setup_logging(config)
    logger.info("=" * 50)
    logger.info("AI伙伴Web服务启动")
    logger.info("=" * 50)

    # 检查API密钥
    api_key = config.get("api", {}).get("api_key", "")
    if not api_key:
        logger.warning("未配置API密钥，某些功能可能受限")

    # 创建对话管理器
    _chat_manager = ChatManager(config)

    # 创建记忆系统
    memory_config = config.get("memory_system", {})
    memory_type = memory_config.get("type", "simple")

    if memory_type == "full":
        logger.info("使用完整记忆系统")
        _memory_system = FullMemorySystem(config)
    else:
        logger.info("使用简单记忆系统")
        short_term_config = config.get("short_term_memory", {})
        max_turns = short_term_config.get("max_turns", 12)
        persist_file = short_term_config.get("persist_file", "data/short_term_memory.json")
        _memory_system = SimpleMemorySystem(max_turns=max_turns, persist_file=persist_file)

    # 将记忆系统设置到对话管理器
    _chat_manager.set_memory_system(_memory_system)

    # 设置状态更新回调
    from .routes.websocket import manager as ws_manager
    global _status_manager
    _status_manager = ws_manager

    def status_update_callback(status: dict = None):
        """状态更新回调函数"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果status为None，则获取当前状态
                if status is None:
                    from .routes.system import get_system_status_internal
                    status = get_system_status_internal(light=True)
                # 使用call_soon_threadsafe来安全地调度异步任务
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast_status(status),
                    loop
                )
        except Exception as e:
            logger.error(f"广播状态更新失败: {e}")

    _memory_system.set_status_update_callback(status_update_callback)
    logger.info("状态更新回调已设置")

    # 初始化命令管理器
    command_config = config.get("commands", {})
    command_prefix = command_config.get("prefix", "/")
    _command_manager = CommandManager(prefix=command_prefix)
    _command_manager.register_default_commands(_memory_system)

    logger.info("服务初始化完成")

    yield

    # 关闭时清理
    logger.info("服务关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="AI伙伴",
        description="AI伙伴记忆系统 - Web服务",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from .routes import chat, memory, system, websocket, rollback
    app.include_router(chat.router, prefix="/api", tags=["对话"])
    app.include_router(memory.router, prefix="/api", tags=["记忆"])
    app.include_router(system.router, prefix="/api", tags=["系统"])
    app.include_router(rollback.router, prefix="/api", tags=["回溯"])
    app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

    # 静态文件服务（前端）
    static_path = Path(__file__).parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # 主页路由
    @app.get("/", response_class=FileResponse)
    async def index():
        """返回前端页面"""
        index_path = Path(__file__).parent.parent / "static" / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "AI伙伴 API服务已启动", "docs": "/docs"}

    # 健康检查
    @app.get("/health")
    async def health():
        """健康检查端点"""
        return {"status": "ok"}

    return app
