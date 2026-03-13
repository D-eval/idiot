import dashscope
from http import HTTPStatus
# 若使用新加坡地域的模型，请取消以下注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
input_text = "衣服的质量杠杠的"
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=input_text,
)

if resp.status_code == HTTPStatus.OK:
    print(resp)


resp['output']['embeddings'][0]['embedding']