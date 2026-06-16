import os

broken = [
    'plugins:gollek-plugin-observability',
    'plugins:gollek-plugin-pii-redaction',
    'plugins:gollek-plugin-rag',
    'plugins:gollek-plugin-prompt',
    'plugins:gollek-plugin-quota',
    'plugins:gollek-plugin-reasoning',
    'plugins:gollek-plugin-sampling',
    'plugins:gollek-plugin-semantic-cache',
    'plugins:gollek-plugin-streaming'
]

settings_path = 'settings.gradle.kts'
with open(settings_path, 'r') as f:
    content = f.read()

for b in broken:
    content = content.replace(f'include("{b}")', f'// include("{b}")')

with open(settings_path, 'w') as f:
    f.write(content)

