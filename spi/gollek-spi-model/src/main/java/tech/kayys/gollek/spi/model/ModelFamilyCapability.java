package tech.kayys.gollek.spi.model;

/**
 * Capabilities advertised by a detachable model-family extension.
 */
public enum ModelFamilyCapability {
    CAUSAL_LM,
    MASKED_LM,
    ENCODER,
    DECODER,
    TOKENIZER,
    CHAT_TEMPLATE,
    EMBEDDING,
    VISION,
    AUDIO,
    MULTIMODAL,
    TRAINING,
    DIRECT_SAFETENSOR_INFERENCE,
    GGUF,
    ONNX
}
