import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers.models.qwen2_my.modeling_qwen2 import Qwen2ForCausalLM_my


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2ForCausalLM_my.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

embedding_layer = model.get_input_embeddings()


def semantic_entropy(logits, topk=200):

    probs = torch.softmax(logits, dim=-1)

    topk_probs, _ = torch.topk(probs, topk)

    topk_probs = topk_probs / topk_probs.sum()

    entropy = -(topk_probs * torch.log(topk_probs + 1e-12)).sum()

    return entropy.item()


# 读取数据
with open("opendataset.json") as f:
    dataset = json.load(f)


results = []

entropy_list = []
label_list = []

for sample in tqdm(dataset):

    prompt = sample["user"]

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    inputs_embeds = embedding_layer(model_inputs["input_ids"]).to(device)

    outputs, hidden_state_all_layers = model(inputs_embeds=inputs_embeds)

    logits = outputs[0][-1]

    entropy = semantic_entropy(logits)

    sample["entropy"] = entropy

    results.append(sample)

    entropy_list.append(entropy)
    label_list.append(sample["label"])


# 保存json
with open("dataset_with_entropy.json","w") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)


# 计算ROC
fpr, tpr, thresholds = roc_curve(label_list, entropy_list)

roc_auc = auc(fpr, tpr)

print("AUROC:", roc_auc)


# 画图
plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc.png")

plt.show()