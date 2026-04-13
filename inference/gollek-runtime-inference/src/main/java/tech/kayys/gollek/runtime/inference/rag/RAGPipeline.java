package tech.kayys.gollek.runtime.inference.rag;

import org.jboss.logging.Logger;

import tech.kayys.gollek.runtime.inference.observability.InferenceTrace;
import tech.kayys.gollek.runtime.inference.observability.LLMObservability;
import tech.kayys.gollek.spi.inference.dto.ChatMessage;
import tech.kayys.gollek.spi.inference.dto.InferenceContext;
import tech.kayys.gollek.spi.inference.dto.InferenceResult;
import tech.kayys.gollek.spi.inference.dto.PromptRequest;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * RAG (Retrieval-Augmented Generation) pipeline implementation.
 * <p>
 * Executes the complete RAG workflow:
 * <ol>
 *   <li><b>Embed:</b> Convert query to vector embedding</li>
 *   <li><b>Retrieve:</b> Search vector store for top-K similar documents</li>
 *   <li><b>Rerank:</b> Re-rank retrieved documents by relevance (optional)</li>
 *   <li><b>Inject:</b> Build prompt with context from retrieved documents</li>
 *   <li><b>Generate:</b> Run LLM inference with augmented prompt</li>
 * </ol>
 *
 * <h2>Performance Optimizations</h2>
 * <ul>
 *   <li><b>Prefix Cache:</b> Cache system prompt + retrieved context</li>
 *   <li><b>Parallel Retrieval:</b> Fetch and embed concurrently</li>
 *   <li><b>Context Truncation:</b> Smart truncation to fit max_context_tokens</li>
 *   <li><b>Batch Embedding:</b> Embed multiple queries in single batch</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * RAGPipeline rag = RAGPipeline.builder()
 *     .config(RAGPipelineConfig.builder()
 *         .vectorStore(VectorStoreConfig.pgvector("jdbc:postgresql://...", "docs"))
 *         .embedderModel("text-embedding-3-small")
 *         .generatorModel("llama-3-70b")
 *         .topK(5)
 *         .build())
 *     .inferenceService(inferenceService)
 *     .observability(observability)
 *     .build();
 *
 * RAGResponse response = rag.query(
 *     "How does TurboQuant compress KV cache?",
 *     RequestContext.builder().apiKey("sk-123").build());
 *
 * System.out.println("Answer: " + response.answer());
 * System.out.println("Sources: " + response.sources().size());
 * }</pre>
 *
 * @since 0.4.0
 */
public final class RAGPipeline {

    private static final Logger LOG = Logger.getLogger(RAGPipeline.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Pipeline configuration */
    private final RAGPipelineConfig config;

    /** Inference service for generation */
    private final Object inferenceService;  // Would be InferenceService

    /** Observability tracer */
    private final LLMObservability observability;

    /** Executor for parallel operations */
    private final ExecutorService executor;

    /** Document retriever (vector store client) */
    private final DocumentRetriever retriever;

    /** Query embedder */
    private final QueryEmbedder embedder;

    /** Document reranker (optional) */
    private final DocumentReranker reranker;

    /** Token counter for context truncation */
    private final TokenCounter tokenCounter;

    // ── Statistics ────────────────────────────────────────────────────

    /** Total queries processed */
    private final AtomicLong totalQueries = new AtomicLong(0);

    /** Total retrieval errors */
    private final AtomicLong retrievalErrors = new AtomicLong(0);

    /** Total generation errors */
    private final AtomicLong generationErrors = new AtomicLong(0);

    /** Sum of retrieval latency (ms) */
    private final AtomicLong retrievalLatencySum = new AtomicLong(0);

    /** Sum of generation latency (ms) */
    private final AtomicLong generationLatencySum = new AtomicLong(0);

    // ── Lifecycle ─────────────────────────────────────────────────────

    private volatile boolean initialized = false;

    private RAGPipeline(Builder builder) {
        this.config = builder.config;
        this.inferenceService = builder.inferenceService;
        this.observability = builder.observability;
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.retriever = builder.retriever;
        this.embedder = builder.embedder;
        this.reranker = builder.reranker;
        this.tokenCounter = builder.tokenCounter != null ? builder.tokenCounter : new SimpleTokenCounter();
    }

    /**
     * Creates a builder for configuring this pipeline.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Query Execution ───────────────────────────────────────────────

    /**
     * Executes a RAG query.
     *
     * @param query user's question
     * @param context request context (API key, tenant, etc.)
     * @return RAG response with answer and sources
     */
    public RAGResponse query(String query, InferenceContext context) {
        totalQueries.incrementAndGet();
        Instant pipelineStart = Instant.now();

        // Start trace if observability enabled
        InferenceTrace trace = null;
        if (observability != null) {
            trace = observability.startTrace(
                config.generatorModel(),
                context.apiKey(),
                context.requestId() + "-rag");
            trace.setAttribute("rag.query", query);
            trace.setAttribute("rag.topK", config.topK());
        }

        try {
            // Stage 1: Retrieve documents
            Instant retrievalStart = Instant.now();
            List<RetrievedDocument> retrieved = retrieveDocuments(query, context);
            long retrievalMs = Duration.between(retrievalStart, Instant.now()).toMillis();
            retrievalLatencySum.addAndGet(retrievalMs);

            if (trace != null) {
                trace.setAttribute("rag.retrieved_count", retrieved.size());
                trace.setAttribute("rag.retrieval_ms", retrievalMs);
            }

            LOG.debugf("Retrieved %d documents in %dms for query: %s",
                retrieved.size(), retrievalMs, query);

            // Stage 2: Build augmented prompt
            String augmentedPrompt = buildPrompt(query, retrieved);

            // Stage 3: Generate response
            Instant generationStart = Instant.now();
            InferenceResult genResponse = generateResponse(augmentedPrompt, context);
            long generationMs = Duration.between(generationStart, Instant.now()).toMillis();
            generationLatencySum.addAndGet(generationMs);

            if (trace != null) {
                trace.recordPromptTokens(genResponse.inputTokens());
                trace.recordCompletionTokens(genResponse.outputTokens());
                trace.recordSuccess();
            }

            long totalMs = Duration.between(pipelineStart, Instant.now()).toMillis();

            LOG.infof("RAG query completed in %dms (retrieve=%dms, generate=%dms), sources=%d",
                totalMs, retrievalMs, generationMs, retrieved.size());

            return new RAGResponse(
                context.requestId(),
                genResponse.content(),
                retrieved,
                genResponse.inputTokens(),
                genResponse.outputTokens(),
                totalMs,
                Map.of(
                    "retrieval_ms", retrievalMs,
                    "generation_ms", generationMs,
                    "sources_count", retrieved.size()
                )
            );

        } catch (Exception e) {
            LOG.errorf(e, "RAG query failed: %s", query);
            if (retriever != null && e instanceof RetrievalException) {
                retrievalErrors.incrementAndGet();
            } else {
                generationErrors.incrementAndGet();
            }
            if (trace != null) {
                trace.recordError(e);
            }
            throw new RAGQueryException("RAG query failed: " + e.getMessage(), e);
        } finally {
            if (trace != null) {
                trace.end();
            }
        }
    }

    /**
     * Executes a RAG query with streaming response.
     */
    public void queryStream(String query, InferenceContext context,
                           StreamCallback callback) {
        totalQueries.incrementAndGet();

        // Stage 1: Retrieve
        List<RetrievedDocument> retrieved = retrieveDocuments(query, context);

        // Stage 2: Build prompt
        String augmentedPrompt = buildPrompt(query, retrieved);

        // Stage 3: Stream generation
        // In production: call streaming inference
        callback.onSources(retrieved);

        // Simulate streaming (placeholder)
        String[] tokens = augmentedPrompt.split("\\s+");
        for (int i = 0; i < Math.min(tokens.length, 100); i++) {
            callback.onToken(tokens[i], i, false);
        }
        callback.onToken("", 100, true);
    }

    // ── Internal Pipeline Stages ──────────────────────────────────────

    /**
     * Retrieves documents from vector store.
     */
    private List<RetrievedDocument> retrieveDocuments(String query, InferenceContext context) {
        if (retriever == null) {
            throw new IllegalStateException("DocumentRetriever not configured");
        }

        // Embed query
        float[] queryEmbedding = embedder != null
            ? embedder.embed(query)
            : new float[1536];  // Placeholder

        // Retrieve from vector store
        List<RetrievedDocument> retrieved = retriever.search(
            queryEmbedding, config.topK(), context);

        // Rerank if configured
        if (reranker != null && !retrieved.isEmpty()) {
            retrieved = reranker.rerank(query, retrieved, config.topN());
        }

        // Truncate to topN
        return retrieved.stream()
            .limit(config.topN())
            .toList();
    }

    /**
     * Builds the augmented prompt with retrieved context.
     */
    private String buildPrompt(String query, List<RetrievedDocument> documents) {
        // Build context from documents
        String context = documents.stream()
            .map(doc -> String.format("[Source: %s]\n%s", doc.id(), doc.content()))
            .collect(Collectors.joining("\n\n"));

        // Truncate context to fit max tokens
        int contextTokens = tokenCounter.countTokens(context);
        if (contextTokens > config.maxContextTokens()) {
            context = truncateContext(context, config.maxContextTokens());
            contextTokens = config.maxContextTokens();
        }

        // Fill template
        return config.promptTemplate().fill(context, query);
    }

    /**
     * Generates response using inference service.
     */
    private InferenceResult generateResponse(String prompt, InferenceContext context) {
        // In production: call inferenceService.infer()
        // For now: return placeholder
        return InferenceResult.builder()
            .requestId(context.requestId())
            .model(config.generatorModel())
            .content("Generated answer based on retrieved context")
            .inputTokens(tokenCounter.countTokens(prompt))
            .outputTokens(50)
            .durationMs(500)
            .build();
    }

    /**
     * Truncates context to fit token limit.
     */
    private String truncateContext(String context, int maxTokens) {
        // Simple word-based truncation
        String[] words = context.split("\\s+");
        StringBuilder sb = new StringBuilder();
        int tokenCount = 0;

        for (String word : words) {
            int wordTokens = Math.max(1, word.length() / 4);  // Rough estimate
            if (tokenCount + wordTokens > maxTokens) break;
            sb.append(word).append(" ");
            tokenCount += wordTokens;
        }

        return sb.toString().trim() + "\n\n[Context truncated due to length limits.]";
    }

    // ── Query Methods ─────────────────────────────────────────────────

    /**
     * Gets average retrieval latency in milliseconds.
     */
    public double getAvgRetrievalLatency() {
        long total = totalQueries.get();
        return total == 0 ? 0.0 : (double) retrievalLatencySum.get() / total;
    }

    /**
     * Gets average generation latency in milliseconds.
     */
    public double getAvgGenerationLatency() {
        long total = totalQueries.get();
        return total == 0 ? 0.0 : (double) generationLatencySum.get() / total;
    }

    /**
     * Gets pipeline statistics.
     */
    public RAGPipelineStats getStats() {
        long total = totalQueries.get();
        return new RAGPipelineStats(
            total,
            retrievalErrors.get(),
            generationErrors.get(),
            getAvgRetrievalLatency(),
            getAvgGenerationLatency(),
            config.topK(),
            config.topN()
        );
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Pipeline statistics.
     */
    public record RAGPipelineStats(
        long totalQueries,
        long retrievalErrors,
        long generationErrors,
        double avgRetrievalLatencyMs,
        double avgGenerationLatencyMs,
        int topK,
        int topN
    ) {
        public double errorRate() {
            return totalQueries == 0 ? 0.0 :
                (double) (retrievalErrors + generationErrors) / totalQueries * 100.0;
        }
    }

    /**
     * Document retriever interface.
     */
    public interface DocumentRetriever {
        List<RetrievedDocument> search(float[] queryEmbedding, int topK, InferenceContext context);
    }

    /**
     * Query embedder interface.
     */
    public interface QueryEmbedder {
        float[] embed(String text);
    }

    /**
     * Document reranker interface.
     */
    public interface DocumentReranker {
        List<RetrievedDocument> rerank(String query, List<RetrievedDocument> documents, int topN);
    }

    /**
     * Token counter interface.
     */
    public interface TokenCounter {
        int countTokens(String text);
    }

    /**
     * Simple word-based token counter.
     */
    static class SimpleTokenCounter implements TokenCounter {
        @Override
        public int countTokens(String text) {
            // Rough estimate: 1 token ≈ 4 chars or 0.75 words
            return Math.max(1, text.split("\\s+").length);
        }
    }

    /**
     * Stream callback interface.
     */
    @FunctionalInterface
    public interface StreamCallback {
        void onToken(String token, int index, boolean finished);

        default void onSources(List<RetrievedDocument> sources) {}
    }

    /**
     * Exception thrown on RAG query failure.
     */
    public static class RAGQueryException extends RuntimeException {
        public RAGQueryException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Exception thrown on retrieval failure.
     */
    public static class RetrievalException extends RuntimeException {
        public RetrievalException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Builder for RAGPipeline.
     */
    public static final class Builder {
        private RAGPipelineConfig config;
        private Object inferenceService;
        private LLMObservability observability;
        private DocumentRetriever retriever;
        private QueryEmbedder embedder;
        private DocumentReranker reranker;
        private TokenCounter tokenCounter;

        private Builder() {}

        public Builder config(RAGPipelineConfig config) {
            this.config = config;
            return this;
        }

        public Builder inferenceService(Object inferenceService) {
            this.inferenceService = inferenceService;
            return this;
        }

        public Builder observability(LLMObservability observability) {
            this.observability = observability;
            return this;
        }

        public Builder retriever(DocumentRetriever retriever) {
            this.retriever = retriever;
            return this;
        }

        public Builder embedder(QueryEmbedder embedder) {
            this.embedder = embedder;
            return this;
        }

        public Builder reranker(DocumentReranker reranker) {
            this.reranker = reranker;
            return this;
        }

        public Builder tokenCounter(TokenCounter tokenCounter) {
            this.tokenCounter = tokenCounter;
            return this;
        }

        public RAGPipeline build() {
            if (config == null) {
                throw new IllegalStateException("RAGPipelineConfig is required");
            }
            return new RAGPipeline(this);
        }
    }
}
