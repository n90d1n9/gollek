import re
import os

replacements = [
    ("runner/gguf/gollek-gguf-core/src/main/java/tech/kayys/gollek/gguf/runtime/GgufRuntimeProfile.java", "hints.getHiddenSize()", "hints.hiddenSize()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTTokenizer.java", "coreTokenizer.getEosTokenId()", "coreTokenizer.eosTokenId()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTTokenizer.java", "coreTokenizer.getVocabSize()", "coreTokenizer.vocabSize()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTGemmaNativeRunner.java", "entry.getModelType()", "entry.modelType()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTContainerParser.java", "e.getModelType()", "e.modelType()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTContainerParser.java", "m.getModelType()", "m.modelType()"),
    ("runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTContainerParser.java", "entry.getModelType()", "entry.modelType()"),
]

for file, old, new in replacements:
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        content = content.replace(old, new)
        with open(file, 'w') as f:
            f.write(content)

