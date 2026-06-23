import re
import os

with open("spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/model-config.md", "r") as f:
    content = f.read()

blocks = re.findall(r"```java\n// (.*?)\n(.*?)```", content, re.DOTALL)
for filename, code in blocks:
    filename = filename.strip()
    if filename == "ModelConfig.java":
        path = "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/ModelConfig.java"
    elif filename in ["ArchitectureMapper.java", "GgufMetadataMapper.java"]:
        path = "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/mapper/" + filename
    elif filename in ["JsonConfigMerger.java", "RopeParameterMerger.java", "ModelConfigLoader.java"]:
        path = "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/loader/" + filename
    else:
        continue
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("// " + filename + "\n" + code)
        print(f"Wrote {path}")
