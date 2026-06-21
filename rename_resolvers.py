import os

files_to_update = [
    "ui/gollek-cli/src/test/java/tech/kayys/gollek/cli/commands/DirectSafetensorRoutePolicyTest.java",
    "ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/RunCommand.java",
    "ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/CliAlternateRuntimeResolver.java",
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route/AlternateRuntimeResolver.java",
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route/DirectSafetensorRoutePolicy.java"
]

for file_path in files_to_update:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r") as f:
        content = f.read()
    
    content = content.replace("AlternateRuntimeResolver", "LocalModelResolver")
    content = content.replace("CliAlternateRuntimeResolver", "CliLocalModelResolver")
    
    with open(file_path, "w") as f:
        f.write(content)

# Rename files
os.rename(
    "ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/CliAlternateRuntimeResolver.java",
    "ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/CliLocalModelResolver.java"
)
os.rename(
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route/AlternateRuntimeResolver.java",
    "runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/route/LocalModelResolver.java"
)
