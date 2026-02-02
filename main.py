"""
AI伙伴主程序
提供命令行交互界面
"""

import sys
import logging
from utils import load_config, setup_logging
from chat import ChatManager
from memory import SimpleMemorySystem, FullMemorySystem

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    if not config:
        print("配置文件加载失败，请检查 config/config.yaml 或 config/config_local.yaml")
        sys.exit(1)

    # 配置日志
    setup_logging(config)

    logger.info("=" * 50)
    logger.info("AI伙伴系统启动")
    logger.info("=" * 50)

    # 检查API密钥
    api_key = config.get("api", {}).get("api_key", "")
    if not api_key:
        print("错误：请在配置文件中设置智谱API的密钥 (api.api_key)")
        print("建议：创建 config/config_local.yaml 文件，并在其中填入API密钥")
        sys.exit(1)

    # 创建对话管理器
    chat_manager = ChatManager(config)

    # 创建记忆系统
    memory_config = config.get("memory_system", {})
    memory_type = memory_config.get("type", "simple")

    if memory_type == "full":
        # 使用完整记忆系统（包含向量数据库和Agent）
        logger.info("使用完整记忆系统")
        memory_system = FullMemorySystem(config)
    else:
        # 使用简单记忆系统（仅短期记忆）
        logger.info("使用简单记忆系统")
        short_term_config = config.get("short_term_memory", {})
        max_turns = short_term_config.get("max_turns", 12)
        memory_system = SimpleMemorySystem(max_turns=max_turns)

    # 将记忆系统设置到对话管理器
    chat_manager.set_memory_system(memory_system)

    # 测试API连接
    print("正在测试API连接...")
    if not chat_manager.test_api_connection():
        print("错误：API连接失败，请检查API密钥和网络连接")
        sys.exit(1)
    print("API连接成功！\n")

    # 交互循环
    print("=" * 50)
    print("AI伙伴已启动！输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    print()

    while True:
        try:
            # 获取用户输入
            try:
                user_input = input("你: ").strip()
            except (EOFError, OSError, BrokenPipeError) as e:
                logger.warning(f"输入流中断: {e}")
                print("\n检测到输入流中断，程序将退出...")
                break

            if not user_input:
                continue

            # 检查退出命令
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n再见！")
                break

            # 发送消息并获取回复
            response = chat_manager.chat(user_input)

            if response:
                print(f"\nAI: {response}\n")
            else:
                print("\nAI: 抱歉，我遇到了一些问题，请稍后再试。\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except BrokenPipeError:
            logger.warning("检测到管道中断")
            print("\n检测到管道中断，程序将退出...")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")
            print(f"\n发生错误: {e}\n")


if __name__ == "__main__":
    main()
