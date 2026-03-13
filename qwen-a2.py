import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

audio_path = "preprocess/segments/Akibare_0000.wav"
text_input = "描述一下这段音频。"


client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key="sk-6f3f7f014a7f47f095ceed6f7e1536e4",
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio(audio_path)

completion = client.chat.completions.create(
    model="qwen3-omni-flash", # 模型为Qwen3-Omni-Flash时，请在非思考模式下运行
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
                {"type": "text", "text": text_input},
            ],
        },
    ],
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if chunk.choices:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
    else:
        print(chunk.usage)
