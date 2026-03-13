from transformers import Qwen2ForCausalLM, AutoTokenizer



model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen3.5-0.8B"

# load the tokenizer and the model
'''
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
'''

model = Qwen2ForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

'''
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
'''
# prepare the model input
prompt = "Give me a short introduction to large language model."

# tokenizer(prompt)

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

embedding_layer = model.get_input_embeddings()

inputs_embeds = embedding_layer(model_inputs['input_ids'])



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