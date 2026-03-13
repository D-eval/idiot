from transformers.models.qwen2_my.modeling_qwen2 import Qwen2ForCausalLM_my
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen3.5-0.8B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the tokenizer and the model
'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
'''

model = Qwen2ForCausalLM_my.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

'''
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
'''
# prepare the model input
prompt = "想象一段声音并描述它？"

# tokenizer(prompt)

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# true_output = "很高兴" # "在一个遥远"
# fake_output = "从前有个傻逼，他不小心吃了一坨屎，然后就"
# text += fake_output

embedding_layer = model.get_input_embeddings()

# while 1:

model_inputs = tokenizer([text], return_tensors="pt").to(device)

inputs_embeds = embedding_layer(model_inputs["input_ids"]).to(device)

outputs, hidden_state_all_layers = model(inputs_embeds=inputs_embeds)

token_prob = torch.topk(outputs[0], k=200, dim=-1)

for i in range(200):
    print(tokenizer.decode([token_prob.indices[0, -1, i].item()]), token_prob.values[0, -1, i].item())

token_id = outputs[0][0, -1, :].argmax()
next_token = tokenizer.decode([token_id])


# 更新text
text += next_token

# end while

'''
hidden_state_all_layers, hidden_state_last_layers = hidden_state_all_layers

prob_distribution_record = []
for (h1,h2) in hidden_state_all_layers:
    # after_attn, after_ffn
    p1 = F.softmax(model.lm_head(h1), dim=-1)
    p2 = F.softmax(model.lm_head(h2), dim=-1)
    p1 = torch.topk(p1, k=5, dim=-1)
    p2 = torch.topk(p2, k=5, dim=-1)
    prob_distribution_record.append((p1,p2))

p_last = F.softmax(model.lm_head(hidden_state_last_layers), dim=-1)
p_last = torch.topk(p_last, k=100, dim=-1)
prob_distribution_record.append(p_last)



# 采样和查字典
token_id = outputs[0][0, -1, :].argmax()
next_token = tokenizer.decode([token_id])
'''


# 问题是 "Who are you?"
# 我靠，原来 tokenizer.decode([40]) 就是 'I'
# 而且 104198 原来就是 "我是" 呀！
# 而且 在 -2,-1 层，观测值都是 40，也就是 'I'
# 在 -3 层，after ffn 的观测值是 82, 也就是 's', after attn 的观测值是 104198, 也就是 '我是'
# 为啥会是 's' 呢？
# 而 -4 层，after ffn 和 after attn 的观测值都是 104198, 也就是 '我是'
# -5 也是，-6, -7 也是
# -8 的 after attn 是 104198 '我是'，after ffn 是 111437，'这里是'
# -9 的 after attn 是 1112 '...'，after ffn 是 151643 '<|endoftext|>'
# -10 都是 151643 '<|endoftext|>'
# -11 也是
# -12 的 after ffn 是 151643 , after attn 是 100110 '一体' (为什么还有这个token啊喂)
# -13 的 after attn 是 108386 , after ffn 是 16 '1'
# -14 的 after attn 是 19793 '示'，after ffn 是 220 ' '
# -15 的 after attn 是 82 's', after ffn 是 82
# -16 的 after attn 是 5615 'Ass' (这tm竟然是token...), after ffn 是 82
# -17 的 after attn 是 79518 '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
# after ffn 是 220
# -18 的 after attn 是 247 ,好像不是 utf-8, after ffn 是 37 'F'
# ...
# 0 的 after attn, after ffn 都是 198 '\n', 1 的 after attn 是 
# 1 的 after ffn 是 7752 'sign', after attn 是 12 '-'
# 2 的 after ffn 是 8339 'yan', after attn 是 66838 '<<<<<<<'
# 3 的 after ffn 是 66838, after attn 是 66838
# 4 也是
# 5 的 after ffn 是 82 's', after attn 是 82
# 6 也是，7 也是
# 8 的 after ffn 是 82， after attn 是 '示'
# 9 的 after ffn 是 220，after attn 是 78 'o'
# 10 的 after ffn 是 37，after attn 是 247
# 10 就是 -18


'''
[    40,   2121,   9707,     48, 100622]]]))
>>> tokenizer.decode([2121])
'As'
>>> tokenizer.decode([9707])
'Hello'
>>> tokenizer.decode([48])
'Q'
>>> tokenizer.decode([100622])
'作为'

我感觉挺好的，但是概率分布是这样的
[9.7280e-01, 1.4963e-02, 9.2506e-03, 7.0217e-04, 4.6594e-04]]]
这基本上只能抽到 40 'I' 吧
'''

# p2[1] 能够
# 究竟可以用 lm_head 观测吗？


'''
import torch
torch.save(embedding_layer.weight, "embedding.pt")
'''

'''
outputs = model(inputs_embeds=inputs_embeds)

max_len = -1 #len(tokenizer)
token_id = outputs[0][0, -1, :max_len].argmax()

next_token = tokenizer.decode([token_id])
'''


'''
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
'''
