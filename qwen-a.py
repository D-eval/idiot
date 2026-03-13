import dashscope
from dashscope import MultiModalConversation
import os
import base64

# 将 ABSOLUTE_PATH/welcome.mp3 替换为本地音频的绝对路径，
# 本地文件的完整路径必须以 file:// 为前缀，以保证路径的合法性，例如：file:///home/images/test.mp3
audio_file_path = "./preprocess/segments/Anemone_0004.wav"

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio(audio_file_path)

messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:;base64,{base64_audio}",
                    "format": "wav",
                },
            },
            {"type": "text", "text": "这段音频在说什么"},
        ],
    },
],

response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-6f3f7f014a7f47f095ceed6f7e1536e4",
    model="qwen3-omni-flash", 
    messages=messages)

print("输出结果为：")
print(response["output"]["choices"][0]["message"]["content"][0]["text"])