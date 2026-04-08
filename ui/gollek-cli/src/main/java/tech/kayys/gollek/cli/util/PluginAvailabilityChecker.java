/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Plugin availability checker for CLI commands.
 * Checks both high-level providers (LLM providers like GGUF, Gemini)
 * and low-level runner plugins (ONNX Runtime, SafeTensor, etc.).
 */
@ApplicationScoped
public class PluginAvailabilityChecker {

    private static final Logger LOG = Logger.getLogger(PluginAvailabilityChecker.class);

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    Instance<RunnerPlugin> runnerPlugins;

    /**
     * Check if any LLM providers are registered.
     */
    public boolean hasProviders() {
        try {
            var providers = providerRegistry.getAllProviders();
            return providers != null && !providers.isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking providers: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Check if a specific provider is available by ID.
     */
    public boolean hasProvider(String providerId) {
        try {
            return providerRegistry.hasProvider(providerId);
        } catch (Exception e) {
            LOG.debugf("Error checking provider %s: %s", providerId, e.getMessage());
            return false;
        }
    }

    /**
     * Check if any runner plugins are discovered.
     * Runner plugins are low-level execution backends (ONNX, GGUF native, etc.)
     */
    public boolean hasRunnerPlugins() {
        try {
            return runnerPlugins != null && !runnerPlugins.isUnsatisfied();
        } catch (Exception e) {
            LOG.debugf("Error checking runner plugins: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get a list of all discovered runner plugin IDs.
     */
    public List<String> getRunnerPluginIds() {
        try {
            return runnerPlugins.stream()
                .map(RunnerPlugin::id)
                .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing runner plugins: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get an error message when no plugins are available at all.
     */
    public String getNoPluginsError() {
        return """
            Error: No inference providers or runner plugins are available.
            
            This typically means no runtime extensions are packaged in this CLI build.
            
            To fix this, ensure at least one of these dependencies is on the classpath:
              - gollek-ext-runner-gguf       (GGUF/llama.cpp runtime)
              - gollek-runner-onnx           (ONNX Runtime)
              - gollek-ext-runner-safetensor  (SafeTensor runtime)
              - gollek-ext-cloud-gemini      (Gemini cloud provider)
              - gollek-ext-cloud-cerebras    (Cerebras cloud provider)
            
            Run 'gollek extensions --all' for details.
            """;
    }

    /**
     * Get an error message when a specific provider is not found.
     */
    public String getProviderNotFoundError(String providerId) {
        StringBuilder sb = new StringBuilder();
        sb.append("Error: Provider '").append(providerId).append("' is not available.\n\n");

        try {
            var available = providerRegistry.getAllProviders();
            if (available != null && !available.isEmpty()) {
                sb.append("Available providers:\n");
                for (var p : available) {
                    sb.append("  - ").append(p.id()).append("\n");
                }
            } else {
                sb.append("No providers are currently available.\n");
            }
        } catch (Exception e) {
            sb.append("Could not list available providers.\n");
        }

        List<String> runners = getRunnerPluginIds();
        if (!runners.isEmpty()) {
            sb.append("\nAvailable runner plugins: ").append(String.join(", ", runners)).append("\n");
        }

        sb.append("\nRun 'gollek extensions --all' for full details.\n");
        return sb.toString();
    }
}
