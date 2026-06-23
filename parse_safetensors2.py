import json
import struct

with open('/Users/bhangun/.gollek/models/blobs/google/gemma-4-E2B-it/google__gemma-4-E2B-it/model.safetensors', 'rb') as f:
    length = struct.unpack('<Q', f.read(8))[0]
    header = f.read(length).decode('utf-8')
    data = json.loads(header)
    for k in data.keys():
        if 'post_attention' in k:
            print("Found:", k)
        if 'post_attention_layernorm' in k:
            print("Found layernorm:", k)
