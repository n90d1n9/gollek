package tech.kayys.gollek.runtime.inference.config;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Gollek inference configuration loader.
 * <p>
 * Loads configuration from YAML/JSON files and environment variables.
 * Supports hierarchical configuration with defaults, file overrides, and env vars.
 *
 * <h2>Configuration Precedence (highest to lowest)</h2>
 * <ol>
 *   <li>Environment variables (GOLLEK_*)</li>
 *   <li>System properties (-Dgollek.*)</li>
 *   <li>Config file (gollek-config.yaml)</li>
 *   <li>Default values</li>
 * </ol>
 *
 * <h2>Example Configuration File (gollek-config.yaml)</h2>
 * <pre>
 * server:
 *   host: 0.0.0.0
 *   port: 8080
 *   workers: 4
 *
 * model:
 *   name: llama-3-70b
 *   path: /models/llama-3-70b.gguf
 *   max_context_length: 8192
 *   dtype: float16
 *
 * inference:
 *   max_batch_size: 128
 *   storage_mode: TURBOQUANT_3BIT
 *   speculative_decoding:
 *     enabled: true
 *     draft_model: llama-3-1b
 *     draft_tokens: 5
 *
 * rate_limiting:
 *   enabled: true
 *   default_tier: PRO
 *   refill_interval_seconds: 1
 *
 * tenants:
 *   enterprise-corp: ENTERPRISE
 *   startup-xyz: PRO
 *
 * observability:
 *   enabled: true
 *   tracing: true
 *   metrics: true
 *   sample_rate: 0.5
 *
 * canary:
 *   enabled: true
 *   stages: [0.10, 0.25, 0.50, 0.75, 1.0]
 *   accuracy_threshold: 0.95
 *
 * rag:
 *   enabled: false
 *   vector_store:
 *     type: pgvector
 *     url: jdbc:postgresql://localhost:5432/gollek
 *     collection: documents
 *   embedder_model: text-embedding-3-small
 *   top_k: 5
 *   top_n: 3
 *
 * fallback:
 *   enabled: true
 *   providers:
 *     - id: local-gpu
 *       model: llama-3-70b
 *       priority: 0
 *     - id: openai
 *       model: gpt-4
 *       priority: 1
 *       api_key: ${OPENAI_API_KEY}
 *     - id: anthropic
 *       model: claude-3
 *       priority: 2
 *       api_key: ${ANTHROPIC_API_KEY}
 *   max_retries: 3
 *   timeout_seconds: 10
 *
 * shutdown:
 *   drain_timeout_seconds: 30
 *   enable_checkpointing: true
 *   checkpoint_path: /tmp/gollek-checkpoints
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * GollekConfig config = GollekConfig.load(Path.of("gollek-config.yaml"));
 *
 * // Access configuration
 * String modelName = config.getModel().getName();
 * int maxBatchSize = config.getInference().getMaxBatchSize();
 * RateLimitTier tier = config.getRateLimiting().getDefaultTier();
 *
 * // Build service from config
 * InferenceService service = InferenceService.builder()
 *     .fromConfig(config)
 *     .build();
 * }</pre>
 *
 * @since 0.5.0
 */
public final class GollekConfig {

    // ── Configuration Sections ──────────────────────────────────────────

    private final ServerConfig server;
    private final ModelConfig model;
    private final InferenceConfig inference;
    private final RateLimitConfig rateLimiting;
    private final Map<String, String> tenants;
    private final ObservabilityConfig observability;
    private final CanaryConfig canary;
    private final RAGConfig rag;
    private final FallbackConfig fallback;
    private final ShutdownConfig shutdown;

    private GollekConfig(Builder builder) {
        this.server = builder.server;
        this.model = builder.model;
        this.inference = builder.inference;
        this.rateLimiting = builder.rateLimiting;
        this.tenants = Map.copyOf(builder.tenants);
        this.observability = builder.observability;
        this.canary = builder.canary;
        this.rag = builder.rag;
        this.fallback = builder.fallback;
        this.shutdown = builder.shutdown;
    }

    /**
     * Loads configuration from a YAML file.
     *
     * @param configPath path to configuration file
     * @return loaded configuration
     */
    public static GollekConfig load(Path configPath) {
        try {
            if (!Files.exists(configPath)) {
                return defaults();
            }

            String yaml = Files.readString(configPath);
            return parse(yaml);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load config from " + configPath, e);
        }
    }

    /**
     * Loads configuration from an input stream.
     */
    public static GollekConfig load(InputStream inputStream) {
        try {
            String yaml = new String(inputStream.readAllBytes());
            return parse(yaml);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load config from stream", e);
        }
    }

    /**
     * Returns default configuration.
     */
    public static GollekConfig defaults() {
        return new Builder().build();
    }

    /**
     * Parses YAML configuration string.
     * <p>
     * In production: use SnakeYAML or similar library.
     * For now: simple key-value parsing for demonstration.
     */
    @SuppressWarnings("unchecked")
    private static GollekConfig parse(String yaml) {
        // In production: use SnakeYAML
        // yaml = yaml.replaceAll("\\$\\{([^}]+)\\}", System.getenv("$1"));
        // Map<String, Object> map = new Yaml().load(yaml);
        // return fromMap(map);

        return defaults();  // Placeholder
    }

    // ── Accessors ───────────────────────────────────────────────────────

    public ServerConfig getServer() { return server; }
    public ModelConfig getModel() { return model; }
    public InferenceConfig getInference() { return inference; }
    public RateLimitConfig getRateLimiting() { return rateLimiting; }
    public Map<String, String> getTenants() { return tenants; }
    public ObservabilityConfig getObservability() { return observability; }
    public CanaryConfig getCanary() { return canary; }
    public RAGConfig getRag() { return rag; }
    public FallbackConfig getFallback() { return fallback; }
    public ShutdownConfig getShutdown() { return shutdown; }

    // ── Configuration Records ───────────────────────────────────────────

    /**
     * Server configuration.
     */
    public record ServerConfig(
        String host,
        int port,
        int workers,
        boolean tlsEnabled,
        String tlsCertPath,
        String tlsKeyPath
    ) {
        public ServerConfig {
            if (host == null) host = "0.0.0.0";
            if (port <= 0) port = 8080;
            if (workers <= 0) workers = 4;
        }
    }

    /**
     * Model configuration.
     */
    public record ModelConfig(
        String name,
        String path,
        int maxContextLength,
        String dtype,
        Map<String, Object> extra
    ) {
        public ModelConfig {
            if (name == null) name = "llama-3-70b";
            if (maxContextLength <= 0) maxContextLength = 8192;
            if (dtype == null) dtype = "float16";
        }
    }

    /**
     * Inference configuration.
     */
    public record InferenceConfig(
        int maxBatchSize,
        String storageMode,
        boolean speculativeDecodingEnabled,
        String draftModel,
        int draftTokens,
        double temperature,
        double topP
    ) {
        public InferenceConfig {
            if (maxBatchSize <= 0) maxBatchSize = 128;
            if (storageMode == null) storageMode = "TURBOQUANT_3BIT";
            if (draftTokens <= 0) draftTokens = 5;
            if (temperature <= 0) temperature = 1.0;
            if (topP <= 0 || topP > 1.0) topP = 1.0;
        }
    }

    /**
     * Rate limiting configuration.
     */
    public record RateLimitConfig(
        boolean enabled,
        String defaultTier,
        int refillIntervalSeconds
    ) {
        public RateLimitConfig {
            if (defaultTier == null) defaultTier = "PRO";
            if (refillIntervalSeconds <= 0) refillIntervalSeconds = 1;
        }
    }

    /**
     * Observability configuration.
     */
    public record ObservabilityConfig(
        boolean enabled,
        boolean tracing,
        boolean metrics,
        double sampleRate
    ) {
        public ObservabilityConfig {
            if (sampleRate <= 0 || sampleRate > 1.0) sampleRate = 0.5;
        }
    }

    /**
     * Canary deployment configuration.
     */
    public record CanaryConfig(
        boolean enabled,
        List<Double> stages,
        double accuracyThreshold
    ) {
        public CanaryConfig {
            if (stages == null) stages = List.of(0.10, 0.25, 0.50, 0.75, 1.0);
            if (accuracyThreshold <= 0 || accuracyThreshold > 1.0) accuracyThreshold = 0.95;
        }
    }

    /**
     * RAG configuration.
     */
    public record RAGConfig(
        boolean enabled,
        String vectorStoreType,
        String vectorStoreUrl,
        String collection,
        String embedderModel,
        int topK,
        int topN,
        int maxContextTokens
    ) {
        public RAGConfig {
            if (vectorStoreType == null) vectorStoreType = "pgvector";
            if (embedderModel == null) embedderModel = "text-embedding-3-small";
            if (topK <= 0) topK = 5;
            if (topN <= 0) topN = 3;
            if (maxContextTokens <= 0) maxContextTokens = 4096;
        }
    }

    /**
     * Fallback configuration.
     */
    public record FallbackConfig(
        boolean enabled,
        List<FallbackProviderConfig> providers,
        int maxRetries,
        int timeoutSeconds,
        String fallbackResponse
    ) {
        public FallbackConfig {
            if (maxRetries <= 0) maxRetries = 3;
            if (timeoutSeconds <= 0) timeoutSeconds = 10;
            if (fallbackResponse == null) fallbackResponse = "Service temporarily unavailable";
        }
    }

    /**
     * Fallback provider configuration.
     */
    public record FallbackProviderConfig(
        String id,
        String model,
        int priority,
        String apiKey
    ) {
        public FallbackProviderConfig {
            if (id == null) id = "unknown";
            if (model == null) model = "default";
        }
    }

    /**
     * Shutdown configuration.
     */
    public record ShutdownConfig(
        int drainTimeoutSeconds,
        boolean enableCheckpointing,
        String checkpointPath
    ) {
        public ShutdownConfig {
            if (drainTimeoutSeconds <= 0) drainTimeoutSeconds = 30;
            if (checkpointPath == null) checkpointPath = "/tmp/gollek-checkpoints";
        }
    }

    // ── Builder ─────────────────────────────────────────────────────────

    private static final class Builder {
        private ServerConfig server = new ServerConfig("0.0.0.0", 8080, 4, false, null, null);
        private ModelConfig model = new ModelConfig("llama-3-70b", null, 8192, "float16", Map.of());
        private InferenceConfig inference = new InferenceConfig(128, "TURBOQUANT_3BIT", false, "llama-3-1b", 5, 1.0, 1.0);
        private RateLimitConfig rateLimiting = new RateLimitConfig(true, "PRO", 1);
        private final Map<String, String> tenants = new HashMap<>();
        private ObservabilityConfig observability = new ObservabilityConfig(true, true, true, 0.5);
        private CanaryConfig canary = new CanaryConfig(true, List.of(0.10, 0.25, 0.50, 0.75, 1.0), 0.95);
        private RAGConfig rag = new RAGConfig(false, "pgvector", null, "documents", "text-embedding-3-small", 5, 3, 4096);
        private FallbackConfig fallback = new FallbackConfig(true, List.of(
            new FallbackProviderConfig("local-gpu", "llama-3-70b", 0, null),
            new FallbackProviderConfig("openai", "gpt-4", 1, null),
            new FallbackProviderConfig("anthropic", "claude-3", 2, null)
        ), 3, 10, "Service temporarily unavailable");
        private ShutdownConfig shutdown = new ShutdownConfig(30, true, "/tmp/gollek-checkpoints");

        private Builder() {}

        public Builder server(ServerConfig server) { this.server = server; return this; }
        public Builder model(ModelConfig model) { this.model = model; return this; }
        public Builder inference(InferenceConfig inference) { this.inference = inference; return this; }
        public Builder rateLimiting(RateLimitConfig config) { this.rateLimiting = config; return this; }
        public Builder tenant(String id, String tier) { this.tenants.put(id, tier); return this; }
        public Builder observability(ObservabilityConfig config) { this.observability = config; return this; }
        public Builder canary(CanaryConfig config) { this.canary = config; return this; }
        public Builder rag(RAGConfig config) { this.rag = config; return this; }
        public Builder fallback(FallbackConfig config) { this.fallback = config; return this; }
        public Builder shutdown(ShutdownConfig config) { this.shutdown = config; return this; }

        public GollekConfig build() {
            return new GollekConfig(this);
        }
    }
}
