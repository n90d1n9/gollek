package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.GollekSdkProvider;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

/**
 * Embedding pipeline that computes dense vector representations of text.
 *
 * <p>Uses the Gollek embedding engine to produce fixed-size {@code float[]} vectors
 * that capture semantic meaning. Vectors can be compared with {@link #similarity}
 * using cosine similarity.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pipeline = new EmbeddingPipeline("all-MiniLM-L6-v2");
 * float[] vector = pipeline.embed("Hello world");
 * float sim = pipeline.similarity("cat", "kitten"); // ~0.85
 * }</pre>
 *
 * @see PipelineFactory
 */
public class EmbeddingPipeline implements Pipeline<String, float[]> {

    private final String modelId;
    private final GollekSdk sdk;

    /**
     * Creates an embedding pipeline for the given model.
     *
     * @param modelId embedding model identifier (e.g. {@code "all-MiniLM-L6-v2"})
     */
    public EmbeddingPipeline(String modelId) {
        this.modelId = modelId;
        this.sdk = resolveSdk();
    }

    /**
     * Embeds a single text string into a dense float vector.
     *
     * @param text the text to embed; must not be {@code null}
     * @return the embedding vector, or an empty array if the response contains no embeddings
     * @throws PipelineException if the embedding request fails
     */
    public float[] embed(String text) {
        try {
            EmbeddingResponse response = sdk.embed(modelId, text);
            if (response.embeddings() != null && !response.embeddings().isEmpty()) {
                return response.embeddings().get(0);
            }
            return new float[0];
        } catch (Exception e) {
            throw new PipelineException("Embedding failed: " + e.getMessage(), e);
        }
    }

    /**
     * Computes the cosine similarity between the embeddings of two texts.
     *
     * <p>Returns a value in {@code [-1.0, 1.0]} where {@code 1.0} means identical
     * direction and {@code 0.0} means orthogonal (unrelated).
     *
     * @param text1 first text
     * @param text2 second text
     * @return cosine similarity score
     * @throws PipelineException if either embedding request fails
     */
    public float similarity(String text1, String text2) {
        float[] v1 = embed(text1);
        float[] v2 = embed(text2);
        return cosineSimilarity(v1, v2);
    }

    /**
     * Delegates to {@link #embed(String)}.
     *
     * @param input the text to embed
     * @return the embedding vector
     * @throws PipelineException if the embedding request fails
     */
    @Override
    public float[] process(String input) {
        return embed(input);
    }

    @Override
    public String task() {
        return "embedding";
    }

    @Override
    public String model() {
        return modelId;
    }

    /**
     * Computes cosine similarity between two vectors.
     * A small epsilon ({@code 1e-8}) is added to the denominator to avoid division by zero.
     *
     * @param a first vector
     * @param b second vector
     * @return cosine similarity in {@code [-1.0, 1.0]}
     */
    private static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length && i < b.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA * normB) + 1e-8f);
    }

    /** Resolves a {@link GollekSdk} via {@link java.util.ServiceLoader}. */
    private GollekSdk resolveSdk() {
        try {
            return java.util.ServiceLoader.load(GollekSdkProvider.class)
                .findFirst().map(p -> {
                    try {
                        return p.create(null);
                    } catch (tech.kayys.gollek.sdk.exception.SdkException e) {
                        return null;
                    }
                }).orElse(null);
        } catch (Exception e) {
            return null;
        }
    }
}
