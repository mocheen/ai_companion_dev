"""
测试新架构的简单脚本
验证自主Agent框架和LLM抽象层
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm import ZhipuLLM
from llm.models import Message, MessageRole, LLMRequest
from agent import BaseAgent, Tool, ToolExecutor
import json


def test_llm_client():
    """测试LLM客户端"""
    print("=" * 50)
    print("测试1: LLM客户端")
    print("=" * 50)

    # 从环境变量或配置文件读取API密钥
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config.get("api", {}).get("api_key", "")
    if not api_key:
        print("❌ 未找到API密钥")
        return False

    # 创建LLM客户端
    llm = ZhipuLLM(api_key=api_key, model="glm-4-flash")

    # 测试简单对话
    print("\n📤 发送测试消息: '你好'")
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="你好")],
        temperature=0.7,
        max_tokens=100
    )

    response = llm.chat(request)
    print(f"📥 收到回复: {response.content}")

    if response.content:
        print("✅ LLM客户端测试通过\n")
        return True
    else:
        print("❌ LLM客户端测试失败\n")
        return False


def test_tool_calling():
    """测试工具调用"""
    print("=" * 50)
    print("测试2: 工具调用")
    print("=" * 50)

    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config.get("api", {}).get("api_key", "")
    if not api_key:
        print("❌ 未找到API密钥")
        return False

    # 创建LLM客户端
    llm = ZhipuLLM(api_key=api_key, model="glm-4.7")

    # 定义工具函数
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        return json.dumps({
            "city": city,
            "temperature": "22°C",
            "condition": "晴天"
        }, ensure_ascii=False)

    # 创建工具执行器
    tool_executor = ToolExecutor()

    # 注册工具
    tool_executor.register_tool(Tool(
        name="get_weather",
        description="获取指定城市的天气信息",
        function=get_weather,
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    ))

    print("\n📤 发送测试消息: '北京今天天气怎么样？'")

    # 创建请求（带工具）
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="北京今天天气怎么样？")],
        tools=tool_executor.get_tools_for_llm(),
        tool_choice="auto"
    )

    response = llm.chat(request)

    if response.has_tool_calls():
        print(f"✅ LLM决定调用工具: {response.tool_calls[0].function.name}")

        # 执行工具
        tool_call = response.tool_calls[0]
        result = tool_executor.execute_tool_call(
            tool_name=tool_call.function.name,
            arguments=tool_call.function.arguments
        )

        print(f"📥 工具执行结果: {result}")

        # 将结果返回给LLM获取最终答案
        messages = [
            Message(role=MessageRole.USER, content="北京今天天气怎么样？"),
            response.to_message(),
            Message(
                role=MessageRole.TOOL,
                content=result,
                tool_call_id=tool_call.id
            )
        ]

        final_request = LLMRequest(
            messages=messages,
            tools=tool_executor.get_tools_for_llm()
        )

        final_response = llm.chat(final_request)
        print(f"📥 LLM最终回复: {final_response.content}")

        print("✅ 工具调用测试通过\n")
        return True
    else:
        print(f"❌ LLM没有调用工具，返回: {response.content}\n")
        return False


def test_agent():
    """测试完整Agent循环"""
    print("=" * 50)
    print("测试3: 自主Agent")
    print("=" * 50)

    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config.get("api", {}).get("api_key", "")
    if not api_key:
        print("❌ 未找到API密钥")
        return False

    # 创建LLM客户端
    llm = ZhipuLLM(api_key=api_key, model="glm-4.7")

    # 定义工具
    def add_numbers(a: int, b: int) -> str:
        """相加两个数字"""
        return json.dumps({"result": a + b}, ensure_ascii=False)

    def multiply_numbers(a: int, b: int) -> str:
        """相乘两个数字"""
        return json.dumps({"result": a * b}, ensure_ascii=False)

    # 创建工具执行器
    tool_executor = ToolExecutor()

    # 注册工具
    tool_executor.register_tool(Tool(
        name="add_numbers",
        description="将两个数字相加",
        function=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    ))

    tool_executor.register_tool(Tool(
        name="multiply_numbers",
        description="将两个数字相乘",
        function=multiply_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    ))

    # 创建Agent
    agent = BaseAgent(
        llm=llm,
        tool_executor=tool_executor,
        system_prompt="你是一个数学助手，可以使用工具进行计算",
        max_iterations=5
    )

    print("\n📤 发送测试消息: '帮我计算 5加3 等于多少，然后乘以2'")

    try:
        result = agent.run("帮我计算 5加3 等于多少，然后乘以2", tool_choice="auto")
        print(f"📥 Agent最终回复: {result}")
        print("✅ Agent测试通过\n")
        return True
    except Exception as e:
        print(f"❌ Agent测试失败: {e}\n")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("AI伙伴 - 新架构测试")
    print("=" * 50 + "\n")

    results = []

    # 测试1: LLM客户端
    results.append(("LLM客户端", test_llm_client()))

    # 测试2: 工具调用
    results.append(("工具调用", test_tool_calling()))

    # 测试3: Agent循环
    results.append(("自主Agent", test_agent()))

    # 总结
    print("=" * 50)
    print("测试总结")
    print("=" * 50)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 所有测试通过！" if all_passed else "⚠️ 部分测试失败"))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
