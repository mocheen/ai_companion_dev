"""
记忆卡片可视化工具
使���Streamlit展示中长期记忆向量数据库的内容
"""

import streamlit as st
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any

from memory.memory_models import EmotionType, DialogueType, LongTermMemoryType
from memory.vector_store import VectorStore


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_emotion(emotion: EmotionType) -> str:
    """格式化情感类型"""
    emotion_map = {
        EmotionType.POSITIVE: "😊 积极",
        EmotionType.NEGATIVE: "😢 消极",
        EmotionType.NEUTRAL: "😐 中性"
    }
    return emotion_map.get(emotion, str(emotion))


def format_dialogue_type(dialogue_type: DialogueType) -> str:
    """格式化对话类型"""
    type_map = {
        DialogueType.CASUAL: "闲聊",
        DialogueType.QUESTION: "问答",
        DialogueType.TASK: "任务",
        DialogueType.EMOTIONAL: "情感交流",
        DialogueType.KNOWLEDGE: "知识分享"
    }
    return type_map.get(dialogue_type, str(dialogue_type))


def format_memory_type(memory_type: LongTermMemoryType) -> str:
    """格式化长期记忆类型"""
    type_map = {
        LongTermMemoryType.PREFERENCE: "偏好",
        LongTermMemoryType.RULE: "规则",
        LongTermMemoryType.EVENT: "事件",
        LongTermMemoryType.KNOWLEDGE: "知识",
        LongTermMemoryType.CHARACTERISTIC: "特征"
    }
    return type_map.get(memory_type, str(memory_type))


def display_medium_term_memory(memory: Dict[str, Any]):
    """展示中期记忆卡片"""
    data = memory["data"]

    st.markdown("---")
    st.subheader(f"📝 {data['topic_summary']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"情感: {format_emotion(EmotionType(data['emotion']))}")
    with col2:
        st.caption(f"类型: {format_dialogue_type(DialogueType(data['dialogue_type']))}")
    with col3:
        st.caption(f"重要性: {data['importance_score']:.2f}")

    st.markdown("**关键信息:**")
    for point in data['key_points']:
        st.markdown(f"- {point}")

    if data.get('topic_tags'):
        tags = ", ".join(data['topic_tags'])
        st.caption(f"标签: {tags}")

    created_at = datetime.fromisoformat(data['created_at'])
    st.caption(f"创建时间: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if data.get('source_message_ids'):
        with st.expander("源消息ID"):
            st.json(data['source_message_ids'])


def display_long_term_memory(memory: Dict[str, Any]):
    """展示长期记忆卡片"""
    data = memory["data"]

    st.markdown("---")
    st.subheader(f"🧠 {data['topic']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"类型: {format_memory_type(LongTermMemoryType(data['memory_type']))}")
    with col2:
        st.caption(f"重要性: {data['importance_score']:.2f}")
    with col3:
        st.caption(f"置信度: {data['confidence_score']:.2f}")

    st.markdown("**摘要:**")
    st.write(data['abstract_summary'])

    created_at = datetime.fromisoformat(data['created_at'])
    st.caption(f"创建时间: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """主函数"""
    st.set_page_config(
        page_title="记忆卡片查看器",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 AI伴侣记忆卡片查看器")

    # 加载配置
    try:
        config = load_config()
        vector_store = VectorStore(config)
        st.success("✅ 成功连接到向量数据库")
    except Exception as e:
        st.error(f"❌ 连接失败: {e}")
        return

    # 侧边栏
    st.sidebar.title("设置")

    memory_type = st.sidebar.radio(
        "选择记忆类型",
        ["中期记忆", "长期记忆"]
    )

    # 获取记忆
    if memory_type == "中期记忆":
        st.header("📝 中期记忆")
        memories = vector_store.get_all_medium_term_memories()
        count = len(memories)
        st.caption(f"共 {count} 条记忆")

        if memories:
            # 排序选项
            sort_by = st.sidebar.selectbox(
                "排序方式",
                ["创建时间（最新）", "创建时间（最旧）", "重要性（高到低）", "重要性（低到高）"]
            )

            # 排序
            if sort_by == "创建时间（最新）":
                memories.sort(key=lambda x: x["data"]["created_at"], reverse=True)
            elif sort_by == "创建时间（最旧）":
                memories.sort(key=lambda x: x["data"]["created_at"])
            elif sort_by == "重要性（高到低）":
                memories.sort(key=lambda x: x["data"]["importance_score"], reverse=True)
            elif sort_by == "重要性（低到高）":
                memories.sort(key=lambda x: x["data"]["importance_score"])

            # 过滤器
            with st.sidebar.expander("过滤器"):
                min_importance = st.slider(
                    "最小重要性",
                    0.0, 1.0, 0.0, 0.1
                )
                emotions = st.multiselect(
                    "情感类型",
                    ["positive", "negative", "neutral"]
                )

            # 应用过滤
            filtered_memories = []
            for mem in memories:
                data = mem["data"]
                if data["importance_score"] >= min_importance:
                    if not emotions or data["emotion"] in emotions:
                        filtered_memories.append(mem)

            st.caption(f"过滤后: {len(filtered_memories)} 条记忆")

            # 展示
            for mem in filtered_memories:
                display_medium_term_memory(mem)
        else:
            st.info("暂无中期记忆")

    else:  # 长期记忆
        st.header("🧠 长期记忆")
        memories = vector_store.get_all_long_term_memories()
        count = len(memories)
        st.caption(f"共 {count} 条记忆")

        if memories:
            # 排序选项
            sort_by = st.sidebar.selectbox(
                "排序方式",
                ["创建时间（最新）", "创建时间（最旧）", "重要性（高到低）", "置信度（高到低）"]
            )

            # 排序
            if sort_by == "创建时间（最新）":
                memories.sort(key=lambda x: x["data"]["created_at"], reverse=True)
            elif sort_by == "创建时间（最旧）":
                memories.sort(key=lambda x: x["data"]["created_at"])
            elif sort_by == "重要性（高到低）":
                memories.sort(key=lambda x: x["data"]["importance_score"], reverse=True)
            elif sort_by == "置信度（高到低）":
                memories.sort(key=lambda x: x["data"]["confidence_score"], reverse=True)

            # 过滤器
            with st.sidebar.expander("过滤器"):
                min_importance = st.slider(
                    "最小重要性",
                    0.0, 1.0, 0.0, 0.1
                )
                memory_types = st.multiselect(
                    "记忆类型",
                    ["preference", "rule", "event", "knowledge", "characteristic"]
                )

            # 应用过滤
            filtered_memories = []
            for mem in memories:
                data = mem["data"]
                if data["importance_score"] >= min_importance:
                    if not memory_types or data["memory_type"] in memory_types:
                        filtered_memories.append(mem)

            st.caption(f"过滤后: {len(filtered_memories)} 条记忆")

            # 展示
            for mem in filtered_memories:
                display_long_term_memory(mem)
        else:
            st.info("暂无长期记忆")


if __name__ == "__main__":
    main()
