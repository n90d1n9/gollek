package tech.kayys.gollek.runtime.inference.model;

import org.jboss.logging.Logger;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Model version manager with A/B testing and canary deployment support.
 * <p>
 * Provides:
 * <ul>
 *   <li><b>Model Versioning:</b> Multiple versions of same model coexist</li>
 *   <li><b>Canary Deployments:</b> Gradual rollout (10% → 50% → 100%)</li>
 *   <li><b>Traffic Splitting:</b> Route requests to different versions</li>
 *   <li><b>Automatic Rollback:</b> Rollback on accuracy degradation</li>
 *   <li><b>A/B Testing:</b> Compare model versions in production</li>
 * </ul>
 *
 * <h2>Canary Deployment Workflow</h2>
 * <pre>
 * 1. Deploy new version (v2) alongside current (v1)
 * 2. Route 10% traffic to v2, 90% to v1
 * 3. Monitor accuracy metrics
 * 4. If accuracy OK → increase to 50%, then 100%
 * 5. If accuracy degraded → rollback to v1
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ModelVersionManager versionManager = ModelVersionManager.builder()
 *     .modelName("llama-3-70b")
 *     .build();
 *
 * // Register versions
 * versionManager.registerVersion("v1", "llama-3-70b-base");
 * versionManager.registerVersion("v2", "llama-3-70b-finetuned");
 *
 * // Start canary deployment (10% traffic to v2)
 * versionManager.startCanary("v2", 0.10);
 *
 * // Route requests
 * String modelId = versionManager.selectVersion("llama-3-70b");
 * // Returns "llama-3-70b-base" 90% of time
 * // Returns "llama-3-70b-finetuned" 10% of time
 *
 * // If canary successful, promote to 100%
 * versionManager.promoteCanary("v2");
 *
 * // If accuracy degraded, rollback
 * versionManager.rollbackCanary("v2");
 * }</pre>
 *
 * @since 0.2.0
 */
public final class ModelVersionManager {

    private static final Logger LOG = Logger.getLogger(ModelVersionManager.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Model name this manager controls */
    private final String modelName;

    /** Canary stages (deployment percentages) */
    private final List<Double> canaryStages;

    /** Accuracy threshold for rollback (0.0 to 1.0) */
    private final double accuracyThreshold;

    // ── Version Registry ──────────────────────────────────────────────

    /** Version ID → ModelVersion */
    private final Map<String, ModelVersion> versions = new ConcurrentHashMap<>();

    /** Current stable version */
    private volatile String currentVersion;

    /** Canary version (if deploying) */
    private volatile String canaryVersion;

    /** Canary traffic percentage (0.0 to 1.0) */
    private volatile double canaryPercentage;

    /** Current canary stage index */
    private volatile int canaryStageIndex;

    // ── Metrics ───────────────────────────────────────────────────────

    /** Total requests routed */
    private final AtomicLong totalRequests = new AtomicLong(0);

    /** Requests routed to canary */
    private final AtomicLong canaryRequests = new AtomicLong(0);

    /** Canary errors */
    private final AtomicLong canaryErrors = new AtomicLong(0);

    /** Stable version errors */
    private final AtomicLong stableErrors = new AtomicLong(0);

    // ── Lifecycle ─────────────────────────────────────────────────────

    private final AtomicBoolean active = new AtomicBoolean(true);

    /** Deployment start time */
    private final Instant deploymentStartTime;

    private ModelVersionManager(Config config) {
        this.modelName = config.modelName;
        this.canaryStages = List.copyOf(config.canaryStages);
        this.accuracyThreshold = config.accuracyThreshold;
        this.deploymentStartTime = Instant.now();
    }

    /**
     * Creates a builder for configuring this manager.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Version Management ────────────────────────────────────────────

    /**
     * Registers a new model version.
     *
     * @param versionId version identifier (e.g., "v1", "v2")
     * @param modelId actual model identifier (e.g., "llama-3-70b-base")
     */
    public void registerVersion(String versionId, String modelId) {
        versions.put(versionId, new ModelVersion(versionId, modelId));
        if (currentVersion == null) {
            currentVersion = versionId;
        }
        LOG.infof("Registered model version %s -> %s", versionId, modelId);
    }

    /**
     * Unregisters a model version.
     */
    public void unregisterVersion(String versionId) {
        versions.remove(versionId);
        if (versionId.equals(currentVersion)) {
            currentVersion = versions.keySet().stream().findFirst().orElse(null);
        }
        if (versionId.equals(canaryVersion)) {
            canaryVersion = null;
            canaryPercentage = 0;
        }
    }

    /**
     * Gets the model ID for a version.
     */
    public String getModelId(String versionId) {
        ModelVersion version = versions.get(versionId);
        return version != null ? version.modelId : null;
    }

    /**
     * Gets all registered versions.
     */
    public Map<String, ModelVersion> getVersions() {
        return Map.copyOf(versions);
    }

    // ── Canary Deployment ─────────────────────────────────────────────

    /**
     * Starts a canary deployment with specified initial percentage.
     *
     * @param versionId version to deploy as canary
     * @param initialPercentage initial traffic percentage (0.0 to 1.0)
     */
    public void startCanary(String versionId, double initialPercentage) {
        if (!versions.containsKey(versionId)) {
            throw new IllegalArgumentException("Version not registered: " + versionId);
        }

        this.canaryVersion = versionId;
        this.canaryPercentage = initialPercentage;
        this.canaryStageIndex = findCanaryStage(initialPercentage);

        LOG.infof("Started canary deployment: version=%s, percentage=%.1f%%, stage=%d",
            versionId, initialPercentage * 100, canaryStageIndex);
    }

    /**
     * Promotes canary to next stage (or 100% if at final stage).
     */
    public void promoteCanary(String versionId) {
        if (!versionId.equals(canaryVersion)) {
            throw new IllegalArgumentException("Version is not canary: " + versionId);
        }

        if (canaryStageIndex >= canaryStages.size() - 1) {
            // Final promotion - canary becomes stable
            currentVersion = canaryVersion;
            canaryVersion = null;
            canaryPercentage = 0;
            canaryStageIndex = 0;
            LOG.infof("Canary promoted to stable: %s", versionId);
        } else {
            // Move to next stage
            canaryStageIndex++;
            canaryPercentage = canaryStages.get(canaryStageIndex);
            LOG.infof("Canary promoted to stage %d: %.1f%%",
                canaryStageIndex, canaryPercentage * 100);
        }
    }

    /**
     * Rolls back canary deployment.
     */
    public void rollbackCanary(String versionId) {
        if (!versionId.equals(canaryVersion)) {
            throw new IllegalArgumentException("Version is not canary: " + versionId);
        }

        LOG.warnf("Rolling back canary deployment: %s", versionId);
        canaryVersion = null;
        canaryPercentage = 0;
        canaryStageIndex = 0;
    }

    /**
     * Checks if canary should be rolled back based on accuracy.
     * <p>
     * Call this periodically with accuracy metrics.
     *
     * @param canaryAccuracy canary version accuracy (0.0 to 1.0)
     * @return true if should rollback
     */
    public boolean shouldRollback(double canaryAccuracy) {
        return canaryAccuracy < accuracyThreshold;
    }

    // ── Traffic Splitting ─────────────────────────────────────────────

    /**
     * Selects which model version to use for a request.
     * <p>
     * If canary is active, routes based on canary percentage.
     * Otherwise, always routes to current stable version.
     *
     * @param modelName model name (for validation)
     * @return selected model ID
     */
    public String selectVersion(String modelName) {
        totalRequests.incrementAndGet();

        // If canary is active, route based on percentage
        if (canaryVersion != null && canaryPercentage > 0) {
            if (ThreadLocalRandom.current().nextDouble() < canaryPercentage) {
                canaryRequests.incrementAndGet();
                return versions.get(canaryVersion).modelId;
            }
        }

        // Default to stable version
        return versions.get(currentVersion).modelId;
    }

    /**
     * Records an error for a version.
     */
    public void recordError(String versionId) {
        if (versionId.equals(canaryVersion)) {
            canaryErrors.incrementAndGet();
        } else if (versionId.equals(currentVersion)) {
            stableErrors.incrementAndGet();
        }
    }

    // ── Query Methods ─────────────────────────────────────────────────

    /**
     * Gets current stable version ID.
     */
    public String getCurrentVersion() {
        return currentVersion;
    }

    /**
     * Gets canary version ID (null if no canary).
     */
    public String getCanaryVersion() {
        return canaryVersion;
    }

    /**
     * Gets canary traffic percentage.
     */
    public double getCanaryPercentage() {
        return canaryPercentage * 100;
    }

    /**
     * Gets canary deployment stage.
     */
    public int getCanaryStage() {
        return canaryStageIndex;
    }

    /**
     * Gets canary error rate.
     */
    public double getCanaryErrorRate() {
        long total = canaryRequests.get();
        return total == 0 ? 0.0 : (double) canaryErrors.get() / total;
    }

    /**
     * Gets stable version error rate.
     */
    public double getStableErrorRate() {
        long stableRequests = totalRequests.get() - canaryRequests.get();
        return stableRequests == 0 ? 0.0 : (double) stableErrors.get() / stableRequests;
    }

    /**
     * Gets deployment statistics.
     */
    public DeploymentStats getStats() {
        return new DeploymentStats(
            modelName,
            currentVersion,
            canaryVersion,
            canaryPercentage * 100,
            canaryStageIndex,
            totalRequests.get(),
            canaryRequests.get(),
            canaryErrors.get(),
            stableErrors.get(),
            getCanaryErrorRate(),
            getStableErrorRate(),
            deploymentStartTime,
            Instant.now()
        );
    }

    // ── Internal Helpers ──────────────────────────────────────────────

    private int findCanaryStage(double percentage) {
        for (int i = 0; i < canaryStages.size(); i++) {
            if (Math.abs(canaryStages.get(i) - percentage) < 0.001) {
                return i;
            }
        }
        return 0;
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Model version metadata.
     */
    public record ModelVersion(
        String versionId,
        String modelId,
        LocalDateTime registeredAt
    ) {
        public ModelVersion(String versionId, String modelId) {
            this(versionId, modelId, LocalDateTime.now());
        }
    }

    /**
     * Deployment statistics.
     */
    public record DeploymentStats(
        String modelName,
        String stableVersion,
        String canaryVersion,
        double canaryPercentage,
        int canaryStage,
        long totalRequests,
        long canaryRequests,
        long canaryErrors,
        long stableErrors,
        double canaryErrorRate,
        double stableErrorRate,
        Instant deploymentStartTime,
        Instant currentTime
    ) {
        /**
         * Deployment duration in seconds.
         */
        public long durationSeconds() {
            return currentTime.getEpochSecond() - deploymentStartTime.getEpochSecond();
        }
    }

    /**
     * Configuration for ModelVersionManager.
     */
    private static final class Config {
        String modelName = "default";
        List<Double> canaryStages = List.of(0.10, 0.25, 0.50, 0.75, 1.0);
        double accuracyThreshold = 0.95;
    }

    /**
     * Builder for ModelVersionManager.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        /**
         * Sets the model name this manager controls.
         */
        public Builder modelName(String modelName) {
            config.modelName = modelName;
            return this;
        }

        /**
         * Sets canary deployment stages (percentages).
         * <p>
         * Default: [0.10, 0.25, 0.50, 0.75, 1.0]
         */
        public Builder canaryStages(List<Double> stages) {
            config.canaryStages = List.copyOf(stages);
            return this;
        }

        /**
         * Sets accuracy threshold for automatic rollback.
         * <p>
         * If canary accuracy drops below this threshold, rollback is recommended.
         * Default: 0.95 (95%)
         */
        public Builder accuracyThreshold(double threshold) {
            config.accuracyThreshold = threshold;
            return this;
        }

        public ModelVersionManager build() {
            return new ModelVersionManager(config);
        }
    }
}
