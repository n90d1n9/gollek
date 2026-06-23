import os
import re

files = [
    "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/loader/JsonConfigMerger.java",
    "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/loader/ModelConfigLoader.java"
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()
    
    # replace log.info(..., arg1, arg2) with log.info(String.format(..., arg1, arg2))
    content = re.sub(r'log\.info\("([^"]+)",([^)]+)\)', r'log.info(String.format("\1",\2))', content)
    content = re.sub(r'log\.warning\("([^"]+)",([^)]+)\)', r'log.warning(String.format("\1",\2))', content)
    
    with open(file, 'w') as f:
        f.write(content)

# Fix ModelFamilyFixtureValidator import again, but correctly this time
val_path = "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/ModelFamilyFixtureValidator.java"
if os.path.exists(val_path):
    with open(val_path, 'r') as f:
        content = f.read()
    if "import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;" not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("import"):
                lines.insert(i, "import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;")
                break
        with open(val_path, 'w') as f:
            f.write('\n'.join(lines))
        print("Fixed imports in ModelFamilyFixtureValidator")

