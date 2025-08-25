from transformers import AutoModelForCausalLM, AutoTokenizer

from mac import MACGemmaDecoderLayer
from titans_pytorch import NeuralMemory

MODEL_NAME = "google/gemma-3-270m-it"
MEMORY_LAYER_IDX = 8

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

memory = NeuralMemory(dim=model.config.hidden_size)

model.model.layers[MEMORY_LAYER_IDX] = MACGemmaDecoderLayer(
    model.config, memory, MEMORY_LAYER_IDX
)


messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
