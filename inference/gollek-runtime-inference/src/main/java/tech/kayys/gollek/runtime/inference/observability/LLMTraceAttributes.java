package tech.kayys.gollek.runtime.inference.observability;

/**
 * LLM-specific attribute keys for OpenTelemetry tracing.
 * <p>
 * These attributes follow the OpenTelemetry semantic conventions for generative AI
 * and extend them with Gollek-specific attributes for production inference serving.
 *
 * <h2>Standard Semantic Conventions</h2>
 * <ul>
 *   <li>gen_ai.system — Model provider (openai, anthropic, gollek-local)</li>
 *   <li>gen_ai.request.model — Model identifier</li>
 *   <li>gen_ai.response.id — Response identifier</li>
 *   <li>gen_ai.usage.input_tokens — Input token count</li>
 *   <li>gen_ai.usage.output_tokens — Output token count</li>
 * </ul>
 *
 * <h2>Gollek Extensions</h2>
 * <ul>
 *   <li>gollek.tenant.id — Tenant identifier</li>
 *   <li>gollek.kv.cache.blocks — KV cache blocks used</li>
 *   <li>gollek.batch.size — Batch size for this request</li>
 *   <li>gollek.compression.ratio — TurboQuant compression ratio</li>
 * </ul>
 *
 * @since 0.2.0
 */
public final class LLMTraceAttributes {

    private LLMTraceAttributes() {}

    // ── Standard Gen AI Attributes ─────────────────────────────────────

    /** Model provider system (openai, anthropic, gollek-local) */
    public static final String GEN_AI_SYSTEM = "gen_ai.system";

    /** Model identifier */
    public static final String GEN_AI_REQUEST_MODEL = "gen_ai.request.model";

    /** Response identifier */
    public static final String GEN_AI_RESPONSE_ID = "gen_ai.response.id";

    /** Input token count */
    public static final String GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens";

    /** Output token count */
    public static final String GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens";

    /** Finish reason (stop, length, content_filter, tool_calls) */
    public static final String GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason";

    // ── Gollek Extensions ─────────────────────────────────────────────

    /** Tenant identifier */
    public static final String GOLLEK_TENANT_ID = "gollek.tenant.id";

    /** Tenant tier (FREE, BASIC, PRO, ENTERPRISE) */
    public static final String GOLLEK_TENANT_TIER = "gollek.tenant.tier";

    /** Request priority level */
    public static final String GOLLEK_REQUEST_PRIORITY = "gollek.request.priority";

    /** KV cache blocks used */
    public static final String GOLLEK_KV_CACHE_BLOCKS = "gollek.kv.cache.blocks";

    /** KV cache memory used (bytes) */
    public static final String GOLLEK_KV_CACHE_BYTES = "gollek.kv.cache.bytes";

    /** Batch size for this request */
    public static final String GOLLEK_BATCH_SIZE = "gollek.batch.size";

    /** Compression ratio (TurboQuant) */
    public static final String GOLLEK_COMPRESSION_RATIO = "gollek.compression.ratio";

    /** Storage mode (full_precision, turboquant_3bit, etc.) */
    public static final String GOLLEK_STORAGE_MODE = "gollek.storage.mode";

    /** Time to first token (TTFT) in milliseconds */
    public static final String GOLLEK_TTFT_MS = "gollek.ttft.ms";

    /** Tokens per second (throughput) */
    public static final String GOLLEK_TOKENS_PER_SEC = "gollek.tokens.per.sec";

    /** Prompt tokens (input after tokenization) */
    public static final String GOLLEK_PROMPT_TOKENS = "gollek.prompt.tokens";

    /** Completion tokens (generated output) */
    public static final String GOLLEK_COMPLETION_TOKENS = "gollek.completion.tokens";

    /** Whether request was rate limited */
    public static final String GOLLEK_RATE_LIMITED = "gollek.rate_limited";

    /** Rate limit tier */
    public static final String GOLLEK_RATE_LIMIT_TIER = "gollek.rate_limit.tier";

    /** Whether request was cancelled */
    public static final String GOLLEK_REQUEST_CANCELLED = "gollek.request.cancelled";

    /** Scheduler queue depth at request time */
    public static final String GOLLEK_QUEUE_DEPTH = "gollek.queue.depth";

    /** Active concurrent requests */
    public static final String GOLLEK_ACTIVE_REQUESTS = "gollek.active.requests";

    /** Model version */
    public static final String GOLLEK_MODEL_VERSION = "gollek.model.version";

    /** GPU device ID (if applicable) */
    public static final String GOLLEK_GPU_DEVICE = "gollek.gpu.device";

    /** GPU memory used (bytes) */
    public static final String GOLLEK_GPU_MEMORY_BYTES = "gollek.gpu.memory.bytes";
}
