from transformers import AutoTokenizer, AutoModelForCausalLM
from torchsummary import summary

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

print(model)