# AI伙伴记忆系统

一个具有三层记忆结构的AI伙伴系统，支持与智谱AI对话，并具有可扩展的记忆系统架构。

## 项目结构

```
ai_companion_dev/
├── config/                 # 配置文件目录
│   ├── config.yaml        # 默认配置文件
│   └── config_local.yaml.example  # 本地配置示例
├── chat/                  # 对话系统模块
│   ├── zhipu_client.py   # 智谱API客��端
│   └── chat_manager.py   # 对话管理器
├── memory/               # 记忆系统模块
│   └── memory_base.py    # 记忆系统基类和简单实现
├── prompts/              # 提示模板目录
│   ├── system_prompt.txt # 系统提示
│   └── user_prompt.txt   # 用户提示模板
├── utils/                # 工具函数
│   └── helpers.py        # 配置加载和日志配置
├── logs/                 # 日志目录
├── main.py              # 主入口文件
├── requirements.txt     # 依赖列表
└── README.md           # 本文件
```

## 功能特点

- **对话系统**：与智谱AI进行对话，支持自定义系统提示
- **解耦设计**：对话系统与记忆系统分离，便于独立开发和测试
- **记忆接口**：预留了完整的记忆系统接口，支持未来扩展
- **灵活配置**：支持通过YAML文件配置各种参数
- **日志记录**：完整的日志记录功能，便于调试和分析

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

1. 复制配置示例文件：
```bash
cp config/config_local.yaml.example config/config_local.yaml
```

2. 编辑 `config/config_local.yaml`，填入你的智谱API密钥：
```yaml
api:
  api_key: "your_api_key_here"  # 替换为你的API密钥
```

## 运行程序

```bash
python main.py
```

## 配置项说明

### API配置
- `api.api_key`: 智谱API的密钥
- `api.model`: 使用的模型名称（默认：glm-4-flash）
- `api.base_url`: API的基础URL

### 短期记忆配置
- `short_term_memory.max_turns`: 短期记忆保存的最大对话轮数（默认：12）

### 对话配置
- `chat.temperature`: 温度参数，控制回复的随机性（0-1，默认：0.7）
- `chat.max_tokens`: 最大生成的token数（默认：2000）
- `chat.timeout`: API请求超时时间（默认：30秒）

### 日志配置
- `logging.level`: 日志级别（DEBUG, INFO, WARNING, ERROR）
- `logging.log_dir`: 日志目录
- `logging.log_file`: 日志文件名

## 提示模板

### 系统提示 (prompts/system_prompt.txt)
定义AI的角色和行为特征。

### 用户提示 (prompts/user_prompt.txt)
用户消息的模板，支持以下占位符：
- `{current_time}`: 当前时间
- `{short_term_memory}`: 短期记忆（最近对话）
- `{long_term_memory}`: 长期记忆
- `{medium_term_memory}`: 中期记忆
- `{user_message}`: 用户当前消息

## 记忆系统接口

当前实现了简单的内存版本记忆系统 `SimpleMemorySystem`，未来可以根据设计文档实现完整的向量数据库版本。

接口方法：
- `get_short_term_memory()`: 获取短期记忆
- `get_long_term_memory(query)`: 基于查询获取长期记忆
- `get_medium_term_memory(query)`: 基于查询获取中期记忆
- `add_conversation(user_msg, assistant_msg)`: 添加对话记录

## 日志

日志文件保存在 `logs/ai_companion.log`，包含：
- INFO级别：系统启动、记忆归档等重要事件
- DEBUG级别：用户消息、API调用详情等详细信息

## 开发路线

当前版本实现了基础的对话功能，后续计划：
1. 实现完整的中期记忆系统（基于ChromaDB）
2. 实现完整的长期记忆系统
3. 使用LangChain Agent实现记忆的自动流转和整理
4. 添加记忆重要性评估机制
5. 优化语义检索算法

## 注意事项

- 请妥善保管你的API密钥，不要将包含真实密钥的配置文件提交到版本控制系统
- 确保网络连接正常，能够访问智谱API服务
- 日志文件会不断增长，定期清理或配置日志轮转

## 许可证

MIT License
