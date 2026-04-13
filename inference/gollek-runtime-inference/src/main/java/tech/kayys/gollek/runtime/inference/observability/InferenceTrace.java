package tech.kayys.gollek.runtime.inference.observability;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Represents a single inference request trace.
 * <p>
 * Captures comprehensive tracing data for an LLM inference request:
 * <ul>
 *   <li><b>Timing:</b> Start time, TTFT, end time, duration</li>
 *   <li><b>Tokens:</b> Input tokens, output tokens, total</li>
 *   <li><b>Performance:</b> Time-to-first-token, tokens/sec, KV cache usage</li>
 *   <li><b>Errors:</b> Error type, message, stack trace</li>
 *   <li><b>Metadata:</b> Tenant, model, priority, storage mode</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * InferenceTrace trace = observability.startTrace("llama-3-70b", "tenant-123", "req-456");
 *
 * try {
 *     // Pre-processing
 *     trace.recordPromptTokens(128);
 *
 *     // Generation loop
 *     trace.recordTTFT(45);  // First token generated
 *     for (int i = 0; i < 256; i++) {
 *         // Generate token...
 *     }
 *     trace.recordCompletionTokens(256);
 *     trace.recordKVCacheUsage(512, 2048);  // blocks, bytes
 *     trace.recordTokensPerSec(850.5);
 *
 *     trace.recordSuccess();
 * } catch (Exception e) {
 *     trace.recordError(e);
 * } finally {
 *     trace.end();
 * }
 * }</pre>
 *
 * @since 0.2.0
 */
public class InferenceTrace {

    // ── Trace Metadata ────────────────────────────────────────────────

    /** Service name */
    private final String serviceName;

    /** Model identifier */
    private final String modelId;

    /** Tenant identifier */
    private final String tenantId;

    /** Request identifier */
    private final String requestId;

    /** Trace start time */
    private final Instant startTime;

    /** Parent observability instance */
    private final LLMObservability observability;

    // ── Timing ────────────────────────────────────────────────────────

    /** Time-to-first-token (milliseconds) */
    private volatile double ttftMs;

    /** Trace end time */
    private volatile Instant endTime;

    // ── Token Counts ──────────────────────────────────────────────────

    /** Prompt (input) tokens */
    private volatile int promptTokens;

    /** Completion (output) tokens */
    private volatile int completionTokens;

    // ── Performance Metrics ───────────────────────────────────────────

    /** Tokens per second throughput */
    private volatile double tokensPerSec;

    /** KV cache blocks used */
    private volatile int kvCacheBlocks;

    /** KV cache memory used (bytes) */
    private volatile long kvCacheBytes;

    // ── Request Metadata ──────────────────────────────────────────────

    /** Request priority */
    private volatile String priority = "normal";

    /** Storage mode */
    private volatile String storageMode = "full_precision";

    /** Compression ratio */
    private volatile double compressionRatio = 1.0;

    /** Batch size */
    private volatile int batchSize = 1;

    /** Rate limited flag */
    private volatile boolean rateLimited;

    /** Rate limit tier */
    private volatile String rateLimitTier;

    /** Cancelled flag */
    private volatile boolean cancelled;

    // ── Result ────────────────────────────────────────────────────────

    /** Whether request succeeded */
    private final AtomicBoolean success = new AtomicBoolean(false);

    /** Error type (if failed) */
    private volatile String errorType;

    /** Error message (if failed) */
    private volatile String errorMessage;

    /** Whether trace has ended */
    private volatile boolean ended;

    /** Custom attributes */
    private final Map<String, Object> attributes = new HashMap<>();

    // ── Constructor ───────────────────────────────────────────────────

    InferenceTrace(String serviceName, String modelId, String tenantId,
                  String requestId, LLMObservability observability) {
        this.serviceName = serviceName;
        this.modelId = modelId;
        this.tenantId = tenantId;
        this.requestId = requestId;
        this.startTime = Instant.now();
        this.observability = observability;
    }

    /**
     * Creates a no-op trace (when tracing is disabled).
     */
    static InferenceTrace noop() {
        return new NoOpInferenceTrace();
    }

    // ── Recording Methods ─────────────────────────────────────────────

    /**
     * Records time-to-first-token.
     *
     * @param ttftMs time to first token in milliseconds
     */
    public void recordTTFT(double ttftMs) {
        this.ttftMs = ttftMs;
    }

    /**
     * Records prompt (input) token count.
     */
    public void recordPromptTokens(int count) {
        this.promptTokens = count;
    }

    /**
     * Records completion (output) token count.
     */
    public void recordCompletionTokens(int count) {
        this.completionTokens = count;
    }

    /**
     * Records throughput in tokens per second.
     */
    public void recordTokensPerSec(double tokensPerSec) {
        this.tokensPerSec = tokensPerSec;
    }

    /**
     * Records KV cache usage.
     *
     * @param blocks number of KV cache blocks used
     * @param bytes memory used in bytes
     */
    public void recordKVCacheUsage(int blocks, long bytes) {
        this.kvCacheBlocks = blocks;
        this.kvCacheBytes = bytes;
    }

    /**
     * Records request success.
     */
    public void recordSuccess() {
        success.set(true);
    }

    /**
     * Records request failure.
     *
     * @param error the exception
     */
    public void recordError(Throwable error) {
        success.set(false);
        this.errorType = error.getClass().getSimpleName();
        this.errorMessage = error.getMessage();
    }

    /**
     * Records rate limiting.
     */
    public void recordRateLimited(String tier) {
        this.rateLimited = true;
        this.rateLimitTier = tier;
    }

    /**
     * Records cancellation.
     */
    public void recordCancelled() {
        this.cancelled = true;
    }

    /**
     * Sets a custom attribute.
     */
    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    // ── Setters for Metadata ──────────────────────────────────────────

    public void setPriority(String priority) { this.priority = priority; }
    public void setStorageMode(String storageMode) { this.storageMode = storageMode; }
    public void setCompressionRatio(double compressionRatio) { this.compressionRatio = compressionRatio; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }

    // ── End Trace ─────────────────────────────────────────────────────

    /**
     * Ends the trace and records metrics.
     */
    public void end() {
        if (ended) return;
        ended = true;
        this.endTime = Instant.now();

        // Record metrics in observability
        if (observability != null) {
            if (success.get()) {
                observability.recordSuccess(tenantId, promptTokens, completionTokens,
                    ttftMs, tokensPerSec);
            } else if (rateLimited) {
                observability.recordRateLimited(tenantId);
            } else if (errorType != null) {
                observability.recordError(tenantId, errorType);
            }
            if (cancelled) {
                observability.recordCancelled();
            }

            observability.endTrace(requestId, this);
        }
    }

    // ── Query Methods ─────────────────────────────────────────────────

    public String getServiceName() { return serviceName; }
    public String getModelId() { return modelId; }
    public String getTenantId() { return tenantId; }
    public String getRequestId() { return requestId; }
    public Instant getStartTime() { return startTime; }
    public Instant getEndTime() { return endTime; }
    public double getTTFT() { return ttftMs; }
    public int getPromptTokens() { return promptTokens; }
    public int getCompletionTokens() { return completionTokens; }
    public int getTotalTokens() { return promptTokens + completionTokens; }
    public double getTokensPerSec() { return tokensPerSec; }
    public int getKVCacheBlocks() { return kvCacheBlocks; }
    public long getKVCacheBytes() { return kvCacheBytes; }
    public String getPriority() { return priority; }
    public String getStorageMode() { return storageMode; }
    public double getCompressionRatio() { return compressionRatio; }
    public int getBatchSize() { return batchSize; }
    public boolean isRateLimited() { return rateLimited; }
    public String getRateLimitTier() { return rateLimitTier; }
    public boolean isCancelled() { return cancelled; }
    public boolean isSuccess() { return success.get(); }
    public String getErrorType() { return errorType; }
    public String getErrorMessage() { return errorMessage; }
    public boolean isEnded() { return ended; }
    public Map<String, Object> getAttributes() { return Map.copyOf(attributes); }

    /**
     * Gets request duration in milliseconds.
     */
    public long getDurationMs() {
        if (endTime == null) {
            return java.time.Duration.between(startTime, Instant.now()).toMillis();
        }
        return java.time.Duration.between(startTime, endTime).toMillis();
    }

    /**
     * Gets trace as a map (for export to OpenTelemetry, Jaeger, etc.).
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put(LLMTraceAttributes.GEN_AI_REQUEST_MODEL, modelId);
        map.put(LLMTraceAttributes.GOLLEK_TENANT_ID, tenantId);
        map.put(LLMTraceAttributes.GEN_AI_RESPONSE_ID, requestId);
        map.put(LLMTraceAttributes.GOLLEK_PROMPT_TOKENS, promptTokens);
        map.put(LLMTraceAttributes.GOLLEK_COMPLETION_TOKENS, completionTokens);
        map.put(LLMTraceAttributes.GOLLEK_TTFT_MS, ttftMs);
        map.put(LLMTraceAttributes.GOLLEK_TOKENS_PER_SEC, tokensPerSec);
        map.put(LLMTraceAttributes.GOLLEK_KV_CACHE_BLOCKS, kvCacheBlocks);
        map.put(LLMTraceAttributes.GOLLEK_KV_CACHE_BYTES, kvCacheBytes);
        map.put(LLMTraceAttributes.GOLLEK_STORAGE_MODE, storageMode);
        map.put(LLMTraceAttributes.GOLLEK_COMPRESSION_RATIO, compressionRatio);
        map.put(LLMTraceAttributes.GOLLEK_BATCH_SIZE, batchSize);
        map.put(LLMTraceAttributes.GOLLEK_RATE_LIMITED, rateLimited);
        map.put(LLMTraceAttributes.GOLLEK_REQUEST_CANCELLED, cancelled);
        map.put("success", success.get());
        map.put("duration_ms", getDurationMs());
        if (errorType != null) {
            map.put("error.type", errorType);
            map.put("error.message", errorMessage);
        }
        map.putAll(attributes);
        return map;
    }

    @Override
    public String toString() {
        return "InferenceTrace[model=%s, tenant=%s, request=%s, tokens=%d, ttft=%.1fms, success=%b]".formatted(
            modelId, tenantId, requestId, getTotalTokens(), ttftMs, success.get());
    }

    // ── No-Op Implementation ─────────────────────────────────────────

    /**
     * No-op trace for when tracing is disabled.
     */
    private static final class NoOpInferenceTrace extends InferenceTrace {
        NoOpInferenceTrace() {
            super("", "", "", "", null);
        }

        @Override public void recordTTFT(double ttftMs) {}
        @Override public void recordPromptTokens(int count) {}
        @Override public void recordCompletionTokens(int count) {}
        @Override public void recordTokensPerSec(double tokensPerSec) {}
        @Override public void recordKVCacheUsage(int blocks, long bytes) {}
        @Override public void recordSuccess() {}
        @Override public void recordError(Throwable error) {}
        @Override public void recordRateLimited(String tier) {}
        @Override public void recordCancelled() {}
        @Override public void end() {}
        @Override public void setAttribute(String key, Object value) {}
    }
}
