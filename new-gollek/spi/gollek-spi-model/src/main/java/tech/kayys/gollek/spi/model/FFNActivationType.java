package tech.kayys.gollek.spi.model;

/**
 * Supported FFN activation functions.
 */
public enum FFNActivationType {
    SILU,   // LLaMA, Mistral
    GELU,   // Gemma, Gemma-2
    RELU,
    GELU_QUICK
}
