import os
import re

directories = [
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route",
    "sdk/gollek-sdk-core/src/main/java/tech/kayys/gollek/sdk/route"
]

def make_public(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Make inner methods public. We'll look for lines starting with exactly 4 or 8 spaces,
    # followed by optional static/final, return type, and method name.
    # Exclude keywords like return, if, for, while, switch.
    content = re.sub(r'^((?: {4}| {8}))(?!public\s|private\s|protected\s|return\s|if\s|for\s|while\s|switch\s|else|catch)([\w<>\[\]\?]+\s+\w+\()', r'\1public \2', content, flags=re.MULTILINE)
    
    # Static methods
    content = re.sub(r'^((?: {4}| {8}))(?!public\s|private\s|protected\s)(static\s+[\w<>\[\]\?]+\s+\w+\()', r'\1public \2', content, flags=re.MULTILINE)

    with open(filepath, 'w') as f:
        f.write(content)

for d in directories:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.java'):
                make_public(os.path.join(root, file))

print("Made methods public.")
