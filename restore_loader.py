import re
import os

with open("spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/model-config.md", "r") as f:
    content = f.read()

blocks = re.findall(r"```java\n// (.*?)\n(.*?)```", content, re.DOTALL)
for filename, code in blocks:
    filename = filename.strip()
    if filename in ["JsonConfigMerger.java", "ModelConfigLoader.java"]:
        path = "spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/loader/" + filename
        with open(path, "w") as f:
            f.write("// " + filename + "\n" + code)
            print(f"Restored {path}")

