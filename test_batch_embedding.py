"""
测试批量embedding功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_models import MediumTermMemory, EmotionType, DialogueType
from memory.vector_store import VectorStore
import yaml

def test_batch_embedding():
    """测试批量embedding功能"""
    print("="*60)
    print("测试批量Embedding功能")
    print("="*60)
    
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 初始化向量存储
    vector_store = VectorStore(config)
    
    # 创建测试记忆
    test_memories = [
        MediumTermMemory(
            topic_summary="测试记忆1",
            key_points=["测试点1", "测试点2"],
            topic_tags=["测试"],
            importance_score=0.8,
            emotion=EmotionType("positive"),
            dialogue_type=DialogueType("casual")
        ),
        MediumTermMemory(
            topic_summary="测试记忆2",
            key_points=["测试点3", "测试点4"],
            topic_tags=["测试"],
            importance_score=0.7,
            emotion=EmotionType("neutral"),
            dialogue_type=DialogueType("knowledge")
        ),
        MediumTermMemory(
            topic_summary="测试记忆3",
            key_points=["测试点5", "测试点6"],
            topic_tags=["测试"],
            importance_score=0.9,
            emotion=EmotionType("positive"),
            dialogue_type=DialogueType("emotional")
        ),
        MediumTermMemory(
            topic_summary="测试记忆4",
            key_points=["测试点7", "测试点8"],
            topic_tags=["测试"],
            importance_score=0.6,
            emotion=EmotionType("neutral"),
            dialogue_type=DialogueType("casual")
        )
    ]
    
    print(f"\n准备批量添加 {len(test_memories)} 条记忆...")
    print("这将使用批量API调用，避免触发速率限制\n")
    
    try:
        # 批量添加
        memory_ids = vector_store.add_medium_term_memories_batch(test_memories)
        
        print(f"✅ 批量添加成功！")
        print(f"生成的记忆ID: {memory_ids}")
        
        # 验证是否添加成功
        count = vector_store.collection_medium.count()
        print(f"\n当前中期记忆总数: {count}")
        
        # 检索测试
        print("\n测试检索功能...")
        results = vector_store.search_medium_term_memories("测试记忆", n_results=3)
        
        print(f"检索到 {len(results)} 条记忆:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['data']['topic_summary']} (距离: {result.get('distance', 0):.4f})")
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！批量Embedding功能正常工作")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_embedding()
    sys.exit(0 if success else 1)
