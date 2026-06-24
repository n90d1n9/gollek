from transformers import AutoConfig
config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
print("Gemma 2 config:", config)
