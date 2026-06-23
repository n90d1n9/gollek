import os
import re

# Mapping of old method names to new method names
replacements = {
    r'\bconfig\.architectures\(\)': 'config.getArchitectures()',
    r'\bconfig\.modelType\(\)': 'config.getModelType()',
    r'\bconfig\.hiddenSize\(\)': 'config.getHiddenSize()',
    r'\bconfig\.numHiddenLayers\(\)': 'config.getNumHiddenLayers()',
    r'\bconfig\.numAttentionHeads\(\)': 'config.getNumAttentionHeads()',
    r'\bconfig\.numKeyValueHeads\(\)': 'config.getNumKeyValueHeads()',
    r'\bconfig\.intermediateSize\(\)': 'config.getIntermediateSize()',
    r'\bconfig\.vocabSize\(\)': 'config.getVocabSize()',
    r'\bconfig\.maxPositionEmbeddings\(\)': 'config.getMaxPositionEmbeddings()',
    r'\bconfig\.rmsNormEps\(\)': 'config.getRmsNormEps()',
    r'\bconfig\.layerNormEps\(\)': 'config.getLayerNormEps()',
    r'\bconfig\.ropeTheta\(\)': 'config.getRopeTheta()',
    r'\bconfig\.ropeScaling\(\)': 'config.getRopeScaling()',
    r'\bconfig\.ropeTypeForLayer\(': 'config.ropeTypeForLayer(', # keep
    r'\bconfig\.layerTypes\(\)': 'config.getLayerTypes()',
    r'\bconfig\.layerType\(': 'config.getLayerType(',
    r'\bconfig\.ropeThetaFull\(\)': 'config.getRopeThetaFull()',
    r'\bconfig\.ropeThetaSliding\(\)': 'config.getRopeThetaSliding()',
    r'\bconfig\.queryPreAttnScalar\(\)': 'config.getQueryPreAttnScalar()',
    r'\bconfig\.attnLogitSoftcapping\(\)': 'config.getAttnLogitSoftcapping()',
    r'\bconfig\.finalLogitSoftcapping\(\)': 'config.getFinalLogitSoftcapping()',
    r'\bconfig\.hiddenAct\(\)': 'config.getHiddenAct()',
    r'\bconfig\.bosTokenId\(\)': 'config.getBosTokenId()',
    r'\bconfig\.eosTokenId\(\)': 'config.getEosTokenId()',
    r'\bconfig\.padTokenId\(\)': 'config.getPadTokenId()',
    r'\bconfig\.tieWordEmbeddings\(\)': 'config.isTieWordEmbeddings()',
    r'\bconfig\.numExpertsPerTok\(\)': 'config.getNumExpertsPerTok()',
    r'\bconfig\.numLocalExperts\(\)': 'config.getNumLocalExperts()',
    r'\bconfig\.moeIntermediateSize\(\)': 'config.getMoeIntermediateSize()',
    r'\bconfig\.visionConfig\(\)': 'config.getVisionConfig()',
    r'\bconfig\.primaryArchitecture\(\)': 'config.getPrimaryArchitecture()',
    r'\bconfig\.ggufArchitectureClassName\(\)': 'config.getGgufArchitectureClassName()'
}

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.java'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()

            new_content = content
            for old, new in replacements.items():
                new_content = re.sub(old, new, new_content)

            if new_content != content:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"Replaced getters in {filepath}")
