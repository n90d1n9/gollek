import os
import glob
import re

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Add imports if needed
    if 'ModelConfig.load' in content or 'ModelConfig.fromDirectory' in content:
        if 'import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;' not in content:
            content = re.sub(r'(import tech.kayys.gollek.spi.model.ModelConfig;)',
                             r'\1\nimport tech.kayys.gollek.spi.model.loader.ModelConfigLoader;', content)
                             
    if 'ModelConfig.fromGgufMetadata' in content:
        if 'import tech.kayys.gollek.spi.model.mapper.GgufMetadataMapper;' not in content:
            content = re.sub(r'(import tech.kayys.gollek.spi.model.ModelConfig;)',
                             r'\1\nimport tech.kayys.gollek.spi.model.mapper.GgufMetadataMapper;', content)

    # Replace ModelConfig.load(path, mapper) with new ModelConfigLoader(mapper).load(path)
    content = re.sub(r'ModelConfig\.load\(([^,]+),\s*([^)]+)\)', r'new ModelConfigLoader(\2).load(\1)', content)
    
    # Replace ModelConfig.fromDirectory(path, mapper) with new ModelConfigLoader(mapper).loadFromDirectory(path)
    content = re.sub(r'ModelConfig\.fromDirectory\(([^,]+),\s*([^)]+)\)', r'new ModelConfigLoader(\2).loadFromDirectory(\1)', content)

    # Replace ModelConfig.fromGgufMetadata(metadata) with new GgufMetadataMapper().fromGgufMetadata(metadata)
    content = re.sub(r'ModelConfig\.fromGgufMetadata\(([^)]+)\)', r'new GgufMetadataMapper().fromGgufMetadata(\1)', content)

    with open(filepath, 'w') as f:
        f.write(content)

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.java'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                if 'ModelConfig.load' in f.read() or 'ModelConfig.fromGgufMetadata' in open(filepath).read() or 'ModelConfig.fromDirectory' in open(filepath).read():
                    replace_in_file(filepath)
                    print(f"Replaced in {filepath}")
