import dashscope
import numpy as np
from dashscope import TextEmbedding

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



# 使用示例
documents = [
    "飞天御剑流",
    "橡胶机关枪",
    "螺旋丸",
    "我要减肥呀"
]
query = "我是路飞！"


query_resp = TextEmbedding.call(
    model="text-embedding-v4",
    input=query,
    dimension=1024
)

query_embedding = query_resp.output['embeddings'][0]['embedding']

# 生成文档向量
doc_resp = TextEmbedding.call(
    model="text-embedding-v4",
    input=documents,
    dimension=1024
)

# 计算相似度
similarities = []
for i, doc_emb in enumerate(doc_resp.output['embeddings']):
    similarity = cosine_similarity(query_embedding, doc_emb['embedding'])
    similarities.append((i, similarity))

# 排序并返回top_k结果
similarities.sort(key=lambda x: x[1], reverse=True)

results = [(documents[i], sim) for i, sim in similarities]

for doc, sim in results:
    print(f"相似度: {sim:.3f}, 文档: {doc}")
