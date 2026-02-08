"""
快速测试修复后的工具调用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from agent import BaseAgent, Tool, ToolExecutor
from llm import ZhipuLLM
from llm.models import Message, MessageRole

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

api_key = config.get("agent_api", {}).get("api_key", "")
if not api_key:
    print("API key not found")
    sys.exit(1)

print("Testing agent with tool calls...")

# 创建LLM
llm = ZhipuLLM(api_key=api_key, model="glm-4.7")

# 定义工具
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建工具执行器
tool_executor = ToolExecutor()

# 注册工具
tool_executor.register_tool(Tool(
    name="get_current_time",
    description="获取当前时间",
    function=get_current_time,
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
))

print("Registered tools:", [t.name for t in tool_executor.get_all_tools()])

# 创建Agent
agent = BaseAgent(
    llm=llm,
    tool_executor=tool_executor,
    system_prompt="你是一个助手，可以使用工具获取信息",
    max_iterations=5
)

# 测试
try:
    result = agent.run("现在几点了？", tool_choice="auto")
    print("\n=== Agent Result ===")
    print(result)
    print("\n=== Test Passed ===")
except Exception as e:
    print(f"\n=== Test Failed ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
