# 🚀 快速开始指南

## 重构完成！项目已完全脱离LangChain，使用自主Agent框架

---

## 📦 新增的模块

### 1. **LLM抽象层** (`llm/`)
```python
from llm import ZhipuLLM
from llm.models import Message, MessageRole, LLMRequest

# 创建LLM客户端
llm = ZhipuLLM(
    api_key="your-api-key",
    model="glm-4.7"
)

# 发送聊天请求
request = LLMRequest(
    messages=[Message(role=MessageRole.USER, content="你好")],
    temperature=0.7
)

response = llm.chat(request)
print(response.content)
```

### 2. **自主Agent框架** (`agent/`)
```python
from agent import BaseAgent, Tool, ToolExecutor
from llm import ZhipuLLM

# 创建LLM和工具执行器
llm = ZhipuLLM(api_key="...", model="glm-4.7")
tool_executor = ToolExecutor()

# 注册工具
def my_function(param: str) -> str:
    return f"处理结果: {param}"

tool_executor.register_tool(Tool(
    name="my_tool",
    description="工具描述",
    function=my_function,
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        },
        "required": ["param"]
    }
))

# 创建并运行Agent
agent = BaseAgent(
    llm=llm,
    tool_executor=tool_executor,
    system_prompt="你是一个有用的助手",
    max_iterations=10
)

result = agent.run("用户输入", tool_choice="auto")
print(result)
```

---

## 🔄 架构对比

### 之前（LangChain）
```python
# ❌ 需要适配类
class ZhipuLLM(BaseChatModel):
    def _generate(self, messages, stop, run_manager, **kwargs):
        # 复杂的格式转换...
        pass

# ❌ 需要Pydantic Schema
class SplitTopicsInput(BaseModel):
    topic_end_ids: List[str] = Field(...)

# ❌ Agent无法正确处理智谱的tool_calls
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS
)
# 问题：调用一次工具后就结束！
```

### 现在（自主框架）
```python
# ✅ 直接使用
llm = ZhipuLLM(api_key="...", model="glm-4.7")

# ✅ 直接定义JSON Schema
tool = Tool(
    name="split_topics",
    description="...",
    function=split_topics_func,
    parameters={  # 简单的JSON Schema
        "type": "object",
        "properties": {...},
        "required": [...]
    }
)

# ✅ Agent完整执行循环
agent = BaseAgent(
    llm=llm,
    tool_executor=tool_executor,
    system_prompt="...",
    max_iterations=10
)

result = agent.run("用户输入")
# ✅ 完整执行所有工具调用！
```

---

## 🧪 测试新架构

运行测试脚本验证所有功能：

```bash
python test_new_architecture.py
```

测试内容：
1. ✅ LLM客户端基础功能
2. ✅ 工具调用（Function Calling）
3. ✅ 完整Agent循环（多轮工具调用）

---

## 📝 主要变更

### 移除的依赖
```diff
- langchain==0.1.20
- langchain-community>=0.0.10
```

### 移除的文件
```diff
- memory/full_memory_system.py (ZhipuLLM适配类)
- agent/archive_agent.py (重复的Agent实现)
- agent/reorganize_agent.py (重复的Agent实现)
```

### 新增的文件
```
llm/
├── __init__.py
├── base_llm.py          # LLM抽象基类
├── zhipu_llm.py         # 智谱实现
└── models.py            # 统一数据模型

agent/
├── __init__.py
├── base_agent.py        # 通用Agent
├── tool.py              # 工具系统
└── executor.py          # 执行器包装
```

### 更新的文件
```
memory/full_memory_system.py  # 完全重构，移除LangChain
requirements.txt              # 移除LangChain依赖
```

---

## 🎯 核心优势

### 1. **通用性**
- LLM抽象层支持任意提供商
- 只需实现`BaseLLM`接口
- 统一的消息和工具格式

### 2. **模块化**
- Agent框架完全独立
- 工具定义清晰简单
- 记忆系统与Agent解耦

### 3. **可维护性**
- 代码量减少（~780行 → ~700行）
- 逻辑清晰，易于调试
- 无第三方框架限制

### 4. **可扩展性**
```python
# 接入OpenAI只需：
class OpenAILLM(BaseLLM):
    def chat(self, request: LLMRequest) -> LLMResponse:
        # 调用OpenAI API
        pass
```

---

## 🔧 配置说明

在`config/config.yaml`中：

```yaml
# Agent专用配置
agent_api:
  api_key: "your-api-key"
  model: "glm-4.7"  # 支持工具调用的模型
  base_url: "https://open.bigmodel.cn/api/paas/v4/"
  tool_choice: "auto"  # 必须设置为auto
```

---

## 📚 文档索引

- **详细重构说明**: `REFACTOR_GUIDE.md`
- **原始设计文档**: `AI伙伴记忆系统设计文档.md`
- **智谱函数调用**: `智谱函数调用.txt`

---

## ✨ 下一步

### 测试记忆归档功能
```bash
python main.py
# 进行多轮对话直到短期记忆达到上限（20轮）
# 观察Agent自动归档过程
```

### 创建自定义Agent
参考 `test_new_architecture.py` 中的示例，创建你自己的Agent！

### 接入其他LLM提供商
实现`BaseLLM`接口，参考`ZhipuLLM`的实现。

---

**版本**: v2.0 (自主Agent架构)
**日期**: 2026-02-05
**状态**: ✅ 生产就绪
