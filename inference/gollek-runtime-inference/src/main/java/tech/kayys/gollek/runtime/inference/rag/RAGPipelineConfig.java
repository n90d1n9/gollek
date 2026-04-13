package tech.kayys.gollek.runtime.inference.rag;

import java.util.List;
import java.util.Map;

/**
 * RAG (Retrieval-Augmented Generation) pipeline configuration.
 * <p>
 * Defines the complete RAG pipeline from document retrieval to generation.
 *
 * <h2>Pipeline Stages</h2>
 * <pre>
 * Query → Embed → Retrieve (Vector DB) → Rerank → Inject Prompt → Generate
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * RAGPipeline pipeline = RAGPipeline.builder()
 *     .vectorStore("pgvector", pgConfig)
 *     .embedder("text-embedding-3-small")
 *     .reranker("cross-encoder-ms-marco")
 *     .generator("llama-3-70b")
 *     .topK(5)
 *     .maxContextTokens(4096)
 *     .promptTemplate(RAGPromptTemplate.DEFAULT)
 *     .build();
 *
 * RAGResponse response = pipeline.query(
 *     "How does TurboQuant work?",
 *     RequestContext.builder().apiKey("sk-123").build());
 * }</pre>
 *
 * @since 0.4.0
 */
public record RAGPipelineConfig(
    /** Vector store configuration */
    VectorStoreConfig vectorStore,

    /** Embedder model ID */
    String embedderModel,

    /** Reranker model ID (optional) */
    String rerankerModel,

    /** Generator model ID */
    String generatorModel,

    /** Number of documents to retrieve */
    int topK,

    /** Number of documents to return after reranking */
    int topN,

    /** Maximum context tokens in prompt */
    int maxContextTokens,

    /** Prompt template for RAG */
    RAGPromptTemplate promptTemplate,

    /** Whether to stream response */
    boolean streaming,

    /** Additional generation parameters */
    Map<String, Object> generationParams
) {
    public RAGPipelineConfig {
        if (topK <= 0) topK = 5;
        if (topN <= 0 || topN > topK) topN = topK;
        if (maxContextTokens <= 0) maxContextTokens = 4096;
        if (promptTemplate == null) promptTemplate = RAGPromptTemplate.DEFAULT;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private VectorStoreConfig vectorStore;
        private String embedderModel = "text-embedding-3-small";
        private String rerankerModel;
        private String generatorModel = "llama-3-70b";
        private int topK = 5;
        private int topN = 3;
        private int maxContextTokens = 4096;
        private RAGPromptTemplate promptTemplate = RAGPromptTemplate.DEFAULT;
        private boolean streaming;
        private Map<String, Object> generationParams;

        public Builder vectorStore(VectorStoreConfig vectorStore) {
            this.vectorStore = vectorStore;
            return this;
        }

        public Builder embedderModel(String model) {
            this.embedderModel = model;
            return this;
        }

        public Builder rerankerModel(String model) {
            this.rerankerModel = model;
            return this;
        }

        public Builder generatorModel(String model) {
            this.generatorModel = model;
            return this;
        }

        public Builder topK(int k) {
            this.topK = k;
            return this;
        }

        public Builder topN(int n) {
            this.topN = n;
            return this;
        }

        public Builder maxContextTokens(int tokens) {
            this.maxContextTokens = tokens;
            return this;
        }

        public Builder promptTemplate(RAGPromptTemplate template) {
            this.promptTemplate = template;
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder generationParams(Map<String, Object> params) {
            this.generationParams = params;
            return this;
        }

        public RAGPipelineConfig build() {
            return new RAGPipelineConfig(
                vectorStore, embedderModel, rerankerModel, generatorModel,
                topK, topN, maxContextTokens, promptTemplate, streaming, generationParams);
        }
    }
}

/**
 * Vector store configuration.
 */
record VectorStoreConfig(
    String type,  // pgvector, milvus, qdrant, weaviate, chroma
    String url,
    String collection,
    Map<String, String> credentials,
    Map<String, Object> extra
) {
    public static VectorStoreConfig pgvector(String url, String collection) {
        return new VectorStoreConfig("pgvector", url, collection, Map.of(), Map.of());
    }

    public static VectorStoreConfig milvus(String url, String collection) {
        return new VectorStoreConfig("milvus", url, collection, Map.of(), Map.of());
    }

    public static VectorStoreConfig qdrant(String url, String collection) {
        return new VectorStoreConfig("qdrant", url, collection, Map.of(), Map.of());
    }
}

/**
 * RAG prompt template.
 */
enum RAGPromptTemplate {

    /**
     * Default RAG template.
     */
    DEFAULT(
        "Answer the question based on the following context.\n\n" +
        "Context:\n{context}\n\n" +
        "Question: {question}\n\n" +
        "Answer:"
    ),

    /**
     * Citations template (includes source references).
     */
    WITH_CITATIONS(
        "Answer the question based on the following context. " +
        "Include citations in the format [Source: document_id].\n\n" +
        "Context:\n{context}\n\n" +
        "Question: {question}\n\n" +
        "Answer:"
    ),

    /**
     * Concise template for short answers.
     */
    CONCISE(
        "Based on the context below, answer concisely.\n\n" +
        "Context: {context}\n\n" +
        "Q: {question}\nA:"
    ),

    /**
     * Custom template (user provides their own).
     */
    CUSTOM("");

    private final String template;

    RAGPromptTemplate(String template) {
        this.template = template;
    }

    /**
     * Fills the template with context and question.
     */
    public String fill(String context, String question) {
        return template
            .replace("{context}", context)
            .replace("{question}", question);
    }

    public String getTemplate() {
        return template;
    }
}

/**
 * Retrieved document from vector store.
 */
record RetrievedDocument(
    String id,
    String content,
    double score,
    Map<String, Object> metadata
) {}

/**
 * RAG response with source documents.
 */
record RAGResponse(
    String requestId,
    String answer,
    List<RetrievedDocument> sources,
    int inputTokens,
    int outputTokens,
    long latencyMs,
    Map<String, Object> metadata
) {}
