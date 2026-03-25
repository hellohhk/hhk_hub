import os

def inject_api_key():
    # 请把下面的字符串替换为你真实的 DeepSeek API Key
    os.environ["DEEPSEEK_API_KEY"] = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"
    print("🔑 API Key 环境变量强行注入成功！")


import requests
import json


def check_deepseek_api(api_key):
    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 发送一个极其简单的请求，仅消耗几个 token
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
    }

    print(f"正在验证 API Key: {api_key[:8]}******")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

        if response.status_code == 200:
            print("✅ 验证成功！API Key 状态正常，余额充足。")
            # print("响应内容:", response.json()['choices'][0]['message']['content'])
        elif response.status_code == 402:
            print("❌ 验证失败：[错误 402] 账户余额不足，请去后台充值。")
        elif response.status_code == 401:
            print("❌ 验证失败：[错误 401] API Key 无效或已过期。")
        elif response.status_code == 429:
            print("⚠️ 验证异常：[错误 429] 触发频率限制（Rate Limit），建议稍后再试。")
        else:
            print(f"❓ 验证异常：返回状态码 {response.status_code}")
            print("错误详情:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"🚫 网络连接失败: {e}")


if __name__ == "__main__":
    # 在这里直接输入你的 API Key 进行测试
    MY_KEY = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"
    check_deepseek_api(MY_KEY)