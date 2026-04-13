package tech.kayys.gollek.plugin.optimization.config;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for optimization plugin configurations.
 * <p>
 * Manages enable/disable state and configuration for all optimization plugins.
 * Provides runtime checks to determine which optimizations should be active.
 * 
 * <h2>Usage</h2>
 * <pre>
 * OptimizationRegistry registry = OptimizationRegistry.getInstance();
 * 
 * // Enable/disable plugins
 * registry.setEnabled("paged-attention", true);
 * registry.setEnabled("flash-attention", false);
 * 
 * // Check if optimization should be applied
 * if (registry.isActive("kv-cache", "cuda")) {
 *     // Apply KV cache optimization for CUDA
 * }
 * 
 * // Get plugin configuration
 * OptimizationConfig config = registry.getConfig("paged-attention");
 * int blockSize = config.getCustomPropertyAsInt("blockSize", 16);
 * </pre>
 * 
 * @since 0.1.0
 */
public class OptimizationRegistry {

    private static final Logger LOG = Logger.getLogger(OptimizationRegistry.class);
    private static volatile OptimizationRegistry instance;

    /** Registered configurations */
    private final Map<String, OptimizationConfig> configurations = new ConcurrentHashMap<>();

    /** Runtime enable/disable state */
    private final Map<String, Boolean> enabledState = new ConcurrentHashMap<>();

    /** Optimization activation listeners */
    private final List<OptimizationStateListener> listeners = new CopyOnWriteArrayList<>();

    private OptimizationRegistry() {
        // Register default configurations
        registerDefaults();
    }

    /**
     * Gets the singleton registry instance.
     */
    public static OptimizationRegistry getInstance() {
        if (instance == null) {
            synchronized (OptimizationRegistry.class) {
                if (instance == null) {
                    instance = new OptimizationRegistry();
                }
            }
        }
        return instance;
    }

    /**
     * Resets the singleton instance (useful for testing).
     */
    public static void reset() {
        instance = null;
    }

    // ── Registration ───────────────────────────────────────────────────

    /**
     * Registers an optimization plugin with default configuration.
     */
    public void register(String pluginId) {
        register(OptimizationConfig.enabled(pluginId));
    }

    /**
     * Registers an optimization plugin with custom configuration.
     */
    public void register(OptimizationConfig config) {
        configurations.put(config.pluginId(), config);
        enabledState.put(config.pluginId(), config.enabled());
        LOG.infof("Registered optimization: %s (enabled=%s, priority=%d)",
            config.pluginId(), config.enabled(), config.priority());
    }

    /**
     * Unregisters an optimization plugin.
     */
    public void unregister(String pluginId) {
        configurations.remove(pluginId);
        enabledState.remove(pluginId);
        LOG.infof("Unregistered optimization: %s", pluginId);
    }

    // ── Enable/Disable ────────────────────────────────────────────────

    /**
     * Enables or disables an optimization plugin at runtime.
     */
    public void setEnabled(String pluginId, boolean enabled) {
        Boolean previous = enabledState.put(pluginId, enabled);
        if (previous == null || previous != enabled) {
            LOG.infof("Optimization %s %s", pluginId, enabled ? "ENABLED" : "DISABLED");
            notifyListeners(pluginId, enabled);
        }
    }

    /**
     * Checks if an optimization is enabled.
     */
    public boolean isEnabled(String pluginId) {
        return enabledState.getOrDefault(pluginId, false);
    }

    /**
     * Checks if an optimization is enabled AND matches the device filter.
     */
    public boolean isActive(String pluginId, String deviceType) {
        OptimizationConfig config = configurations.get(pluginId);
        if (config == null) return false;
        if (!isEnabled(pluginId)) return false;
        return config.matchesDevice(deviceType);
    }

    // ── Configuration Access ──────────────────────────────────────────

    /**
     * Gets the configuration for an optimization plugin.
     */
    public OptimizationConfig getConfig(String pluginId) {
        return configurations.get(pluginId);
    }

    /**
     * Gets all registered optimization configurations.
     */
    public Map<String, OptimizationConfig> getAllConfigs() {
        return Collections.unmodifiableMap(configurations);
    }

    /**
     * Gets all active optimizations for a specific device type.
     */
    public List<OptimizationConfig> getActiveOptimizations(String deviceType) {
        return configurations.values().stream()
            .filter(config -> isEnabled(config.pluginId()))
            .filter(config -> config.matchesDevice(deviceType))
            .sorted(Comparator.comparingInt(OptimizationConfig::priority))
            .toList();
    }

    // ── Listeners ─────────────────────────────────────────────────────

    /**
     * Adds a listener for optimization state changes.
     */
    public void addListener(OptimizationStateListener listener) {
        listeners.add(listener);
    }

    /**
     * Removes a listener.
     */
    public void removeListener(OptimizationStateListener listener) {
        listeners.remove(listener);
    }

    private void notifyListeners(String pluginId, boolean enabled) {
        for (OptimizationStateListener listener : listeners) {
            try {
                listener.onStateChanged(pluginId, enabled);
            } catch (Exception e) {
                LOG.warnf(e, "Error notifying listener of optimization state change: %s", pluginId);
            }
        }
    }

    // ── Default Registration ──────────────────────────────────────────

    private void registerDefaults() {
        // Core optimizations - all enabled by default
        register(OptimizationConfig.builder("paged-attention")
            .priority(10)
            .customProperty("blockSize", "16")
            .customProperty("totalBlocks", "2048")
            .build());

        register(OptimizationConfig.builder("kv-cache")
            .priority(20)
            .customProperty("evictionPolicy", "LRU")
            .customProperty("maxSequenceLength", "32768")
            .build());

        register(OptimizationConfig.builder("continuous-batching")
            .priority(15)
            .customProperty("maxBatchSize", "256")
            .customProperty("scheduleIntervalMs", "10")
            .build());

        register(OptimizationConfig.builder("flash-attention")
            .priority(5)
            .addDeviceFilter("cuda")
            .addDeviceFilter("metal")
            .addDeviceFilter("rocm")
            .customProperty("minSeqLen", "512")
            .build());

        register(OptimizationConfig.builder("prompt-cache")
            .priority(25)
            .customProperty("maxPrompts", "1000")
            .customProperty("ttlMinutes", "60")
            .build());

        register(OptimizationConfig.builder("hybrid-attention")
            .priority(30)
            .addDeviceFilter("cuda")
            .build());

        register(OptimizationConfig.builder("evicpress")
            .priority(35)
            .addDeviceFilter("cuda")
            .addDeviceFilter("metal")
            .build());

        register(OptimizationConfig.builder("perf-mode")
            .priority(40)
            .customProperty("mode", "balanced")
            .build());

        register(OptimizationConfig.builder("prefill-decode")
            .priority(45)
            .addDeviceFilter("cuda")
            .build());

        register(OptimizationConfig.builder("wait-scheduler")
            .priority(50)
            .build());

        register(OptimizationConfig.builder("weight-offload")
            .priority(55)
            .customProperty("offloadTo", "cpu")
            .build());

        register(OptimizationConfig.builder("qlora")
            .priority(60)
            .customProperty("rank", "16")
            .customProperty("alpha", "32")
            .build());

        register(OptimizationConfig.builder("elastic-ep")
            .priority(65)
            .addDeviceFilter("cuda")
            .addDeviceFilter("rocm")
            .build());

        register(OptimizationConfig.builder("fa3")
            .priority(3)
            .addDeviceFilter("cuda")
            .customProperty("minComputeCapability", "90")
            .build());

        register(OptimizationConfig.builder("fa4")
            .priority(2)
            .addDeviceFilter("cuda")
            .addDeviceFilter("metal")
            .customProperty("minComputeCapability", "100")
            .build());
    }

    // ── Nested Types ────────────────────────────────────────────────

    /**
     * Listener for optimization state changes.
     */
    public interface OptimizationStateListener {
        void onStateChanged(String pluginId, boolean enabled);
    }
}
