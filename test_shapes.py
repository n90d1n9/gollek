import json
with open('/Users/bhangun/.gollek/models/blobs/Qwen/Qwen3.6-35B-A3B/Qwen__Qwen3.6-35B-A3B/config.json') as f:
    config = json.load(f)
print("Hidden size:", config['text_config']['hidden_size'])
