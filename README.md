# AI伙伴记忆系统

一个具有三层记忆结构的智能对话系统，支持与智谱AI进行自然对话，并具备完整的记忆管理、归档和检索功能。

## 功能特点

- **三层记忆架构**：短期记忆（最近对话）、中期记忆（结构化记忆卡片）、长期记忆（深度知识）
- **智能记忆流转**：自动将对话内容归档为结构化记忆，并定期重整优化
- **自主Agent框架**：使用自主开发的Agent系统实现记忆归档和重整，不依赖LangChain
- **向量数据库检索**：基于ChromaDB的语义检索，智能匹配相关记忆
- **多种交互方式**：支持命令行交互和Web界面（含WebSocket实时通信）
- **灵活配置**：通过YAML配置文件自定义各项参数
- **完整日志系统**：详细的日志记录，便于调试和分析

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

复制配置示例文件：

```bash
cp config/config.yaml.example config/config.yaml
```

编辑 `config/config.yaml`，填入你的智谱API密钥：

```yaml
api:
  api_key: "your_api_key_here"  # 替换为你的API密钥
  model: "glm-4.7"
```

同时配置Agent专用API密钥（用于记忆归档和重整）：

```yaml
agent_api:
  api_key: "your_api_key_here"  # 替换为你的API密钥
```

### 3. 启动服务

#### 命令行模式

```bash
python main.py
```

启动后即可在命令行中与AI伙伴对话。输入 `/help` 查看可用命令。

#### Web模式

```bash
python run_web.py
```

启动后访问 `http://localhost:5163` 使用Web界面。

#### Docker模式

```bash
docker-compose up -d
```

## 项目结构

```
ai_companion_dev/
├── agent/                 # Agent框架模块
│   ├── base_agent.py     # Agent基类
│   ├── executor.py       # 工具执行器
│   └── tool.py           # 工具定义
├── api/                  # Web服务模块
│   ├── routes/           # API路由
│   │   ├── chat.py       # 对话接口
│   │   ├── memory.py     # 记忆管理接口
│   │   ├── system.py     # 系统接口
│   │   └── websocket.py  # WebSocket接口
│   └── app.py            # FastAPI应用
├── chat/                 # 对话系统模块
│   ├── chat_manager.py   # 对话管理器
│   └── zhipu_client.py   # 智谱API客户端
├── llm/                  # LLM模块
│   ├── base_llm.py       # LLM基类
│   ├── models.py         # 数据模型
│   └── zhipu_llm.py      # 智谱LLM实现
├── memory/               # 记忆系统模块
│   ├── prompts/          # Agent提示模板
│   ├── full_memory_system.py  # 完整记忆系统
│   ├── memory_base.py    # 记忆系统基类
│   ├── memory_models.py  # 记忆数据模型
│   └── vector_store.py   # 向量数据库封装
├── config/               # 配置文件目录
│   └── config.yaml.example  # 配置示例
├── prompts/              # 提示模板目录
│   ├── system_prompt.txt # 系统提示
│   └── user_prompt.txt   # 用户提示模板
├── static/               # Web前端静态文件
│   └── index.html        # Web界面
├── utils/                # 工具函数
│   ├── helpers.py        # 配置加载和日志配置
│   └── rate_limiter.py   # 速率限制
├── data/                 # 数据目录（运行时生成）
│   ├── chromadb/         # 向量数据库
│   └── *.json            # 记忆持久化文件
├── logs/                 # 日志目录
├── main.py              # 命令行主入口
├── run_web.py          # Web服务启动脚本
├── view_memory.py      # 记忆查看工具
├── migrate_chromadb.py # 数据迁移工具
├── command_manager.py  # 命令管理器
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```

## 配置说明

### 核心配置项

#### API配置
```yaml
api:
  api_key: "your_api_key"        # 智谱API密钥
  model: "glm-4.7"               # 使用的模型
  base_url: "https://open.bigmodel.cn/api/paas/v4/"
```

#### Agent配置
```yaml
agent_api:
  api_key: "your_api_key"        # Agent专用API密钥
  model: "glm-4.7"
  tool_choice: "auto"            # 工具选择策略

agent:
  archive:
    max_iterations: 20           # 归档Agent最大迭代次数
  reorganize:
    max_iterations: 40           # 重整Agent最大迭代次数
```

#### 记忆系统配置
```yaml
memory_system:
  type: "full"                   # "simple" 或 "full"

short_term_memory:
  max_turns: 20                  # 短期记忆最大对话轮数
  persist_file: "data/short_term_memory.json"

memory_retrieval:
  medium_term_count: 3           # 检索的中期记忆数量
  long_term_count: 2             # 检索的长期记忆数量
```

#### 向量数据库配置
```yaml
vector_db:
  embedding_type: "api"          # "api" 或 "local"
  embedding_model: "embedding-3"  # 智谱嵌入模型
  persistence_dir: "data/chromadb"
  collection_medium: "medium_term_memories"
  collection_long: "long_term_memories"
```

#### 记忆流转配置
```yaml
memory_flow:
  archive_interval_days: 7       # 中期记忆重整间隔（天）
  flow_state_file: "data/memory_flow_state.json"
```

#### 日志配置
```yaml
logging:
  level: "DEBUG"                 # 日志级别
  log_dir: "logs"
  log_file: "ai_companion.log"
  enable_file: true             # 是否启用文件日志
```

## 使用方法

### 命令行模式

启动后可使用以下命令：

- `/help` - 显示帮助信息
- `/exit` - 退出程序
- `/clear` - 清空短期记忆
- `/archive` - 手动触发记忆归档
- `/reorganize` - 手动触发记忆重整
- `/stats` - 显示记忆统计信息
- `/view` - 查看记忆内容

### Web模式

Web界面提供以下功能：

- 实时对话（WebSocket支持）
- 记忆管理（查看、删除记忆）
- 系统状态监控
- API文档（访问 `/docs`）

### 记忆查看工具

使用 `view_memory.py` 查看和分析记忆内容：

```bash
python view_memory.py
```

## 记忆系统架构

### 三层记忆结构

1. **短期记忆**：保存最近的对话轮数（可配置），用于维持对话上下文
2. **中期记忆**：使用ChromaDB向量数据库存储结构化记忆卡片，支持语义检索
3. **长期记忆**：深度知识和重要信息，同样使用向量数据库存储

### 记忆流转流程

1. 对话内容自动保存到短期记忆
2. 达到一定条件后，Agent自动将对话归档为中期记忆
3. 定期重整中期记忆，优化记忆质量
4. 重要信息升级为长期记忆

### Agent功能

- **归档Agent**：将对话内容提取为结构化记忆卡片
- **重整Agent**：优化和整理现有记忆，去重和合并相似记忆

## 开发路线

- [ ] 优化记忆重要性评估机制
- [ ] 增强语义检索算法
- [ ] 支持多用户会话隔离
- [ ] 添加记忆导出功能
- [ ] 实现记忆可视化界面
- [ ] 支持更多LLM提供商

## 注意事项

- 请妥善保管API密钥，不要将包含真实密钥的配置文件提交到版本控制系统
- 确保网络连接正常，能够访问智谱API服务
- 如使用向量数据库，首次运行会自动下载嵌入模型（如使用本地嵌入）
- 日志文件会不断增长，建议定期清理或配置日志轮转，也可以直接关闭日志文件写入
- 使用Docker部署时，确保正确配置环境变量

## 许可证

MIT License
