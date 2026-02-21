"""
管理员命令管理器
提供统一的命令注册、解析和执行功能
"""

import logging
import asyncio
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from threading import Thread

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """命令执行结果"""
    success: bool
    message: str
    should_exit: bool = False  # 是否应该退出程序


@dataclass
class Command:
    """命令定义"""
    name: str                          # 命令名称 (不含前缀)
    description: str                   # 命令描述
    usage: str                         # 使用方法
    handler: Callable[..., CommandResult]  # 命令处理函数
    aliases: List[str] = None          # 命令别名

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class CommandManager:
    """
    管理员命令管理器

    支持通过配置的前缀（如 "/"）来调用管理员命令
    """

    def __init__(self, prefix: str = "/"):
        """
        初始化命令管理器

        Args:
            prefix: 命令前缀，默认为 "/"
        """
        self.prefix = prefix
        self.commands: Dict[str, Command] = {}
        self._running_tasks: List[Thread] = []  # 跟踪运行中的异步任务

        logger.info(f"命令管理器初始化完成，前缀: '{prefix}'")

    def register_command(self, command: Command):
        """
        注册命令

        Args:
            command: 命令对象
        """
        # 注册主命令名
        self.commands[command.name.lower()] = command

        # 注册别名
        for alias in (command.aliases or []):
            self.commands[alias.lower()] = command

        logger.debug(f"注册命令: {command.name}, 别名: {command.aliases}")

    def parse_input(self, user_input: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        解析用户输入

        Args:
            user_input: 用户输入字符串

        Returns:
            Tuple[是否为命令, 命令名, 参数列表]
        """
        user_input = user_input.strip()

        if not user_input.startswith(self.prefix):
            return False, None, []

        # 移除前缀
        content = user_input[len(self.prefix):].strip()

        if not content:
            return True, "", []

        # 分割命令和参数
        parts = content.split(maxsplit=1)
        command_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        return True, command_name, args

    def execute(self, user_input: str) -> Optional[CommandResult]:
        """
        执行命令

        Args:
            user_input: 用户输入

        Returns:
            如果是命令，返回执行结果；如果不是命令，返回 None
        """
        is_command, command_name, args = self.parse_input(user_input)

        if not is_command:
            return None

        # 空命令（只有前缀）
        if not command_name:
            return CommandResult(
                success=False,
                message=f"请输入命令。使用 {self.prefix}help 查看可用命令。"
            )

        # 查找命令
        command = self.commands.get(command_name)

        if not command:
            return CommandResult(
                success=False,
                message=f"未知命令: {self.prefix}{command_name}。使用 {self.prefix}help 查看可用命令。"
            )

        # 执行命令
        try:
            result = command.handler(*args)
            logger.info(f"执行命令: {self.prefix}{command_name}, 结果: {result.success}")
            return result
        except Exception as e:
            logger.error(f"执行命令 {self.prefix}{command_name} 失败: {e}", exc_info=True)
            return CommandResult(
                success=False,
                message=f"命令执行失败: {e}"
            )

    def get_help_text(self) -> str:
        """获取帮助文本"""
        lines = ["可用命令:"]
        lines.append("-" * 40)

        # 去重（因为别名会注册多次）
        seen_names = set()
        for name, command in self.commands.items():
            if command.name in seen_names:
                continue
            seen_names.add(command.name)

            # 构建命令行
            cmd_line = f"  {self.prefix}{command.name}"
            if command.aliases:
                aliases_str = ", ".join(f"{self.prefix}{a}" for a in command.aliases)
                cmd_line += f" ({aliases_str})"

            lines.append(cmd_line)
            lines.append(f"    {command.description}")
            lines.append(f"    用法: {self.prefix}{command.usage}")
            lines.append("")

        lines.append("-" * 40)
        lines.append(f"提示: 使用 {self.prefix}命令名 来执行命令")

        return "\n".join(lines)

    def register_default_commands(self, memory_system=None):
        """
        注册默认的管理员命令

        Args:
            memory_system: 记忆系统实例（可选）
        """
        # 帮助命令
        self.register_command(Command(
            name="help",
            description="显示可用命令列表",
            usage="help",
            handler=lambda: CommandResult(
                success=True,
                message=self.get_help_text()
            ),
            aliases=["h", "?"]
        ))

        # 退出命令
        def handle_exit(*args) -> CommandResult:
            return CommandResult(
                success=True,
                message="再见！",
                should_exit=True
            )

        self.register_command(Command(
            name="exit",
            description="退出程序",
            usage="exit",
            handler=handle_exit,
            aliases=["quit", "q"]
        ))

        # 如果提供了记忆系统，注册记忆相关命令
        if memory_system:
            self._register_memory_commands(memory_system)

    def _register_memory_commands(self, memory_system):
        """注册记忆相关命令"""

        # 查看短期记忆
        def handle_short_memory(*args) -> CommandResult:
            try:
                # 获取短期记忆
                messages = memory_system.get_short_term_memory()

                if not messages:
                    return CommandResult(
                        success=True,
                        message="短期记忆队列为空"
                    )

                # 格式化输出
                lines = [f"短期记忆队列 (共 {len(messages)} 条):"]
                lines.append("=" * 50)

                for msg in messages:
                    time_str = msg.timestamp.strftime("%m-%d %H:%M:%S") if msg.timestamp else "??"
                    role_display = "用户" if msg.role == "user" else "AI"
                    # 截断过长的内容
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    lines.append(f"[{time_str}] {role_display}: {content}")

                lines.append("=" * 50)
                lines.append(f"队列容量: {len(messages)}/{getattr(memory_system, 'max_messages', '?')}")

                return CommandResult(
                    success=True,
                    message="\n".join(lines)
                )

            except Exception as e:
                logger.error(f"获取短期记忆失败: {e}", exc_info=True)
                return CommandResult(
                    success=False,
                    message=f"获取短期记忆失败: {e}"
                )

        self.register_command(Command(
            name="short",
            description="查看短期记忆队列",
            usage="short",
            handler=handle_short_memory,
            aliases=["stm", "memory"]
        ))

        # 手动触发归档
        def handle_archive(*args) -> CommandResult:
            try:
                # 检查是否有归档方法
                if not hasattr(memory_system, '_trigger_archive_async'):
                    return CommandResult(
                        success=False,
                        message="当前记忆系统不支持手动归档"
                    )

                # 检查短期记忆是否为空
                messages = memory_system.get_short_term_memory()
                if not messages:
                    return CommandResult(
                        success=True,
                        message="短期记忆队列为空，无需归档"
                    )

                # 异步触发归档
                memory_system._trigger_archive_async()

                return CommandResult(
                    success=True,
                    message=f"已触发短期记忆归档任务（异步执行中...）\n当前短期记忆: {len(messages)} 条"
                )

            except Exception as e:
                logger.error(f"触发归档失败: {e}", exc_info=True)
                return CommandResult(
                    success=False,
                    message=f"触发归档失败: {e}"
                )

        self.register_command(Command(
            name="archive",
            description="手动触发短期记忆归档（异步）",
            usage="archive",
            handler=handle_archive
        ))

        # 手动触发重整
        def handle_reorganize(*args) -> CommandResult:
            try:
                # 检查是否有重整方法
                if not hasattr(memory_system, '_trigger_reorganize_async'):
                    return CommandResult(
                        success=False,
                        message="当前记忆系统不支持手动重整"
                    )

                # 异步触发重整
                memory_system._trigger_reorganize_async()

                return CommandResult(
                    success=True,
                    message="已触发中期记忆重整任务（异步执行中...）\n这可能需要几分钟时间完成"
                )

            except Exception as e:
                logger.error(f"触发重整失败: {e}", exc_info=True)
                return CommandResult(
                    success=False,
                    message=f"触发重整失败: {e}"
                )

        self.register_command(Command(
            name="reorganize",
            description="手动触发中期记忆重整（异步）",
            usage="reorganize",
            handler=handle_reorganize,
            aliases=["reorg"]
        ))

        # 查看记忆系统状态
        def handle_status(*args) -> CommandResult:
            try:
                lines = ["记忆系统状态:"]
                lines.append("-" * 40)

                # 短期记忆
                short_messages = memory_system.get_short_term_memory()
                max_messages = getattr(memory_system, 'max_messages', '未知')
                lines.append(f"短期记忆: {len(short_messages)}/{max_messages} 条")

                # 中期记忆（如果支持）
                if hasattr(memory_system, 'vector_store'):
                    try:
                        medium_memories = memory_system.vector_store.get_all_medium_term_memories()
                        lines.append(f"中期记忆: {len(medium_memories)} 条")
                    except Exception:
                        lines.append("中期记忆: 无法获取")

                    try:
                        long_memories = memory_system.vector_store.get_all_long_term_memories()
                        lines.append(f"长期记忆: {len(long_memories)} 条")
                    except Exception:
                        lines.append("长期记忆: 无法获取")

                # 记忆类型
                memory_type = type(memory_system).__name__
                lines.append(f"记忆系统类型: {memory_type}")

                lines.append("-" * 40)

                return CommandResult(
                    success=True,
                    message="\n".join(lines)
                )

            except Exception as e:
                logger.error(f"获取记忆状态失败: {e}", exc_info=True)
                return CommandResult(
                    success=False,
                    message=f"获取记忆状态失败: {e}"
                )

        self.register_command(Command(
            name="status",
            description="查看记忆系统状态",
            usage="status",
            handler=handle_status,
            aliases=["stat", "info"]
        ))
