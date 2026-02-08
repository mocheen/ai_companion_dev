"""
智谱AI Embedding API测试脚本
用于测试embedding-2和embedding-3模型的可用性
"""

import requests
import json
import time

# API配置
API_KEY = "24574694451e4109a93b500d1af68688.pVBQOOdyF2sKfLTR"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_embedding_model(model_name):
    """
    测试指定的embedding模型

    Args:
        model_name: 模型名称（如embedding-2或embedding-3）
    """
    url = f"{BASE_URL}embeddings"
    
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "input": ["这是一段测试文本，用于验证智谱AI的embedding API是否可用。"]
    }
    
    print(f"\n请求URL: {url}")
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应时间: {elapsed_time:.2f}秒")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功！")
            print(f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"\n向量维度: {len(embedding)}")
                print(f"前5个向量值: {embedding[:5]}")
            
            if "usage" in result:
                usage = result["usage"]
                print(f"\nToken使用情况:")
                print(f"  - prompt_tokens: {usage.get('prompt_tokens', 0)}")
                print(f"  - total_tokens: {usage.get('total_tokens', 0)}")
            
            return True
        else:
            print(f"\n❌ 失败！")
            print(f"错误信息: {response.text}")
            
            if response.status_code == 429:
                print(f"\n⚠️  429错误 - 请求过多或权限不足")
                print(f"可能的原因:")
                print(f"  1. API Key没有embedding模型的调用权限")
                print(f"  2. 超过了速率限制（并发请求数过多）")
                print(f"  3. 账户等级限制")
            
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n❌ 请求超时（>30秒）")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求异常: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        return False

def test_batch_embedding(model_name):
    """
    测试批量embedding请求

    Args:
        model_name: 模型名称
    """
    url = f"{BASE_URL}embeddings"
    
    print(f"\n{'='*60}")
    print(f"测试批量请求: {model_name}")
    print(f"{'='*60}")
    
    test_texts = [
        "第一段测试文本",
        "第二段测试文本",
        "第三段测试文本"
    ]
    
    payload = {
        "model": model_name,
        "input": test_texts
    }
    
    print(f"\n批量请求数量: {len(test_texts)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 批量请求成功！")
            print(f"返回向量数量: {len(result.get('data', []))}")
            
            if "usage" in result:
                usage = result["usage"]
                print(f"Token使用: {usage.get('total_tokens', 0)}")
            
            return True
        else:
            print(f"❌ 批量请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 批量请求异常: {e}")
        return False

def test_chat_api():
    """
    测试chat API作为对比
    """
    url = f"{BASE_URL}chat/completions"
    
    print(f"\n{'='*60}")
    print(f"测试Chat API（作为对比）")
    print(f"{'='*60}")
    
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat API可用！")
            content = result["choices"][0]["message"]["content"]
            print(f"响应内容: {content}")
            return True
        else:
            print(f"❌ Chat API失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat API异常: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("智谱AI Embedding API 测试脚本")
    print("="*60)
    print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
    print(f"基础URL: {BASE_URL}")
    
    results = {}
    
    # 测试Chat API
    results['chat'] = test_chat_api()
    
    # 等待一下
    time.sleep(2)
    
    # 测试embedding-2
    results['embedding-2'] = test_embedding_model("embedding-2")
    
    # 等待一下
    time.sleep(2)
    
    # 测试embedding-3
    results['embedding-3'] = test_embedding_model("embedding-3")
    
    # 等待一下
    time.sleep(2)
    
    # 测试批量请求
    results['embedding-3-batch'] = test_batch_embedding("embedding-3")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} : {status}")
    
    # 给出建议
    print(f"\n{'='*60}")
    print("建议")
    print(f"{'='*60}")
    
    if not results.get('embedding-2') and not results.get('embedding-3'):
        print("\n❌ 所有embedding模型都不可用")
        print("\n可能的原因:")
        print("1. API Key没有embedding模型的调用权限")
        print("2. 需要在智谱AI控制台开通embedding服务")
        print("3. 账户余额不足或需要升级套餐")
        print("\n建议操作:")
        print("- 登录 https://open.bigmodel.cn/")
        print("- 检查API Key的权限设置")
        print("- 查看账户余额和套餐信息")
        print("- 联系智谱AI客服咨询embedding API权限")
    elif results.get('embedding-3') and not results.get('embedding-2'):
        print("\n✅ embedding-3可用，但embedding-2不可用")
        print("建议: 使用embedding-3模型")
    elif results.get('embedding-2') and not results.get('embedding-3'):
        print("\n✅ embedding-2可用，但embedding-3不可用")
        print("建议: 使用embedding-2模型")
    else:
        print("\n✅ 所有测试都通过！API配置正常。")

if __name__ == "__main__":
    main()
