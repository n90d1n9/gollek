/*
 * Gollek Inference Engine — Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * EmbeddingPooling.java
 * ──────────────────────
 * Stateless embedding pooling and normalisation utilities.
 *
 * Extracted from gollek-safetensor-engine:EmbeddingEngine.
 * Used by SafeTensor, ONNX, and any future embedding backends.
 */
package tech.kayys.gollek.embedding;

/**
 * Stateless utility methods for embedding pooling and normalisation.
 *
 * <p>
 * These methods operate on raw float arrays produced by any transformer backend.
 * They are intentionally dependency-free (no CDI, no Vert.x) to be reusable in
 * batch processing, testing, and non-CDI contexts.
 */
public final class EmbeddingPooling {

    private EmbeddingPooling() {}

    // ─────────────────────────────── Pooling ──────────────────────────────────

    /**
     * Mean pooling: average all token positions.
     *
     * @param hidden     flat float array of shape {@code [seqLen × hiddenSize]}
     * @param seqLen     number of tokens
     * @param hiddenSize embedding dimension
     * @return pooled vector of length {@code hiddenSize}
     */
    public static float[] meanPool(float[] hidden, int seqLen, int hiddenSize) {
        float[] pooled = new float[hiddenSize];
        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < hiddenSize; d++) {
                pooled[d] += hidden[t * hiddenSize + d];
            }
        }
        float inv = 1.0f / seqLen;
        for (int d = 0; d < hiddenSize; d++) {
            pooled[d] *= inv;
        }
        return pooled;
    }

    /**
     * CLS pooling: take the first token ([CLS]) hidden state at position 0.
     *
     * @param hidden     flat float array of shape {@code [seqLen × hiddenSize]}
     * @param hiddenSize embedding dimension
     * @return CLS token vector of length {@code hiddenSize}
     */
    public static float[] clsPool(float[] hidden, int hiddenSize) {
        float[] result = new float[hiddenSize];
        System.arraycopy(hidden, 0, result, 0, hiddenSize);
        return result;
    }

    /**
     * Pool hidden states according to a {@link PoolingStrategy}.
     *
     * @param hidden     flat float array {@code [seqLen × hiddenSize]}
     * @param seqLen     number of tokens
     * @param hiddenSize embedding dimension
     * @param strategy   pooling strategy to apply
     * @return pooled embedding vector
     */
    public static float[] pool(float[] hidden, int seqLen, int hiddenSize, PoolingStrategy strategy) {
        return switch (strategy) {
            case MEAN_POOL -> meanPool(hidden, seqLen, hiddenSize);
            case CLS_POOL  -> clsPool(hidden, hiddenSize);
        };
    }

    // ────────────────────────── Normalisation ───────────────────────────────

    /**
     * L2-normalise an embedding vector so its Euclidean norm equals 1.
     *
     * <p>
     * After normalisation, cosine similarity equals dot product, which is
     * the most efficient retrieval metric for vector databases.
     *
     * @param vec input vector (modified in-place return copy)
     * @return new L2-normalised vector; returns {@code vec} unchanged if norm ≈ 0
     */
    public static float[] l2Normalize(float[] vec) {
        float norm = 0f;
        for (float v : vec) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        if (norm < 1e-8f) {
            return vec;
        }
        float invNorm = 1.0f / norm;
        float[] out = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            out[i] = vec[i] * invNorm;
        }
        return out;
    }

    // ─────────────────────── Similarity metrics ──────────────────────────────

    /**
     * Cosine similarity between two embedding vectors.
     *
     * <p>
     * If both vectors are L2-normalised (via {@link #l2Normalize}), this is
     * equivalent to the dot product and runs in O(n).
     *
     * @param a first embedding (should be L2-normalised for cosine semantics)
     * @param b second embedding (should be L2-normalised for cosine semantics)
     * @return similarity value in [-1.0, 1.0]
     * @throws IllegalArgumentException if dimensions do not match
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Embedding dimension mismatch: " + a.length + " vs " + b.length);
        }
        float dot = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    /**
     * Dot product of two embedding vectors (unnormalised cosine similarity).
     *
     * @param a first vector
     * @param b second vector
     * @return dot product
     */
    public static float dotProduct(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Dimension mismatch: " + a.length + " vs " + b.length);
        }
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
