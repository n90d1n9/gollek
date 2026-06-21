import os
import re

directories = [
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route",
    "sdk/gollek-sdk-core/src/main/java/tech/kayys/gollek/sdk/route"
]

def make_public(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Make inner interfaces, classes, records, enums public
    content = re.sub(r'^( {4})((?:static\s+)?(?:final\s+)?(?:class|record|interface|enum)\s+\w+)', r'\1public \2', content, flags=re.MULTILINE)
    
    # Make methods public (look for '    static' or '    <Type>')
    content = re.sub(r'^( {4})(static\s+[\w<>,\[\]\s]+\s+\w+\()', r'\1public \2', content, flags=re.MULTILINE)
    
    # Make fields public (e.g. static final String, static final int)
    content = re.sub(r'^( {4})(static\s+final\s+[\w<>,\[\]\s]+\s+\w+\s*=)', r'\1public \2', content, flags=re.MULTILINE)

    # Make AUTO public in RunnerRoutePolicy
    content = re.sub(r'^( {4})(final\s+[\w<>,\[\]\s]+\s+\w+\s*=)', r'\1public \2', content, flags=re.MULTILINE)

    # Deduplicate "public public"
    content = content.replace('public public ', 'public ')

    with open(filepath, 'w') as f:
        f.write(content)

for d in directories:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.java'):
                make_public(os.path.join(root, file))

print("Made members public.")
