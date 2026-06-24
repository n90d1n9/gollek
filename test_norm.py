import torch
from transformers import AutoModelForCausalLM, AutoConfig

print("Checking Gemma 2/3/4 normalization")
# Since we don't have the weights locally, we can inspect the HuggingFace transformers code if it's installed.
