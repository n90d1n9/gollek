/*
 * Gollek Inference Engine — Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * PoolingStrategy.java
 * ─────────────────────
 * Enumeration of embedding pooling strategies used by transformer models.
 */
package tech.kayys.gollek.embedding;

/**
 * Embedding pooling strategy for transformer encoder/decoder outputs.
 *
 * <ul>
 *   <li>{@link #MEAN_POOL} — average all token hidden states. Used by nomic-embed,
 *       e5-large, GTE, BGE (LLaMA/Mistral-family encoders).</li>
 *   <li>{@link #CLS_POOL} — take the [CLS] token hidden state at position 0. Used by
 *       BERT, DistilBERT, RoBERTa, XLM-RoBERTa.</li>
 * </ul>
 */
public enum PoolingStrategy {

    /**
     * Mean pooling: average all token positions.
     * <p>
     * {@code hidden[1, seqLen, hiddenSize] → mean over dim=1 → [1, hiddenSize]}
     */
    MEAN_POOL,

    /**
     * CLS pooling: use the first token ([CLS]) hidden state.
     * <p>
     * {@code hidden[1, seqLen, hiddenSize] → hidden[:, 0, :] → [1, hiddenSize]}
     */
    CLS_POOL;

    /**
     * Detect the appropriate pooling strategy for a given model type string.
     *
     * @param modelType the {@code model_type} field from a HuggingFace {@code config.json}
     * @return inferred pooling strategy
     */
    public static PoolingStrategy fromModelType(String modelType) {
        if (modelType == null) return MEAN_POOL;
        return switch (modelType.toLowerCase()) {
            case "bert", "distilbert", "roberta", "xlm-roberta", "albert", "electra" -> CLS_POOL;
            default -> MEAN_POOL;  // LLaMA, Mistral, Qwen, Nomic, etc.
        };
    }
}
