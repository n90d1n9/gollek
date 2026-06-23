import safetensors.torch
import sys

def check_weights(path):
    f = safetensors.torch.load_file(path)
    if "model.layers.0.post_per_layer_input_norm.weight" in f:
        print("post_per_layer_input_norm.weight shape:", f["model.layers.0.post_per_layer_input_norm.weight"].shape)
    if "model.layers.0.per_layer_projection.weight" in f:
        print("per_layer_projection.weight shape:", f["model.layers.0.per_layer_projection.weight"].shape)

check_weights("/Users/bhangun/.gollek/models/blobs/google/gemma-4-E2B-it/google__gemma-4-E2B-it/model-00001-of-00002.safetensors")
