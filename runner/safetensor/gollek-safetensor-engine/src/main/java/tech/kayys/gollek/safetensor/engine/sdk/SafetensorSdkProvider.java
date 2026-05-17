package tech.kayys.gollek.safetensor.engine.sdk;

import io.quarkus.arc.Arc;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.sdk.config.SdkConfig;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.GollekSdkProvider;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.provider.ProviderRegistry;

import java.nio.file.Path;

/**
 * ServiceLoader-discoverable provider that creates a {@link SafetensorGollekSdk}
 * backed by the SafeTensor {@link DirectInferenceEngine}.
 *
 * <p>This provider is registered via
 * {@code META-INF/services/tech.kayys.gollek.sdk.core.GollekSdkProvider}
 * and is discovered automatically by the NLP pipeline factory when the
 * SafeTensor engine is on the classpath.
 *
 * <p>Priority is set to 50 (lower = higher priority) so it takes precedence
 * over remote providers when running locally.
 */
public class SafetensorSdkProvider implements GollekSdkProvider {

    @Override
    public Mode mode() {
        return Mode.LOCAL;
    }

    @Override
    public GollekSdk create(SdkConfig config) throws SdkException {
        try {
            // Create a standalone DirectInferenceEngine for non-CDI contexts
            // In a CDI container (Quarkus), the engine would be injected instead
            DirectInferenceEngine engine = new DirectInferenceEngine();
            ProviderRegistry providerRegistry = resolveProviderRegistry();

            // Resolve model base path from config or defaults
            Path modelBasePath = resolveModelBasePath(config);

            return new SafetensorGollekSdk(engine, modelBasePath, providerRegistry);
        } catch (Exception e) {
            throw new SdkException("Failed to create SafeTensor SDK: " + e.getMessage(), e);
        }
    }

    @Override
    public int priority() {
        // Higher priority than remote providers (default=100)
        return 50;
    }

    private Path resolveModelBasePath(SdkConfig config) {
        // Check for explicit configuration
        if (config != null && config.getGgufBasePath() != null) {
            return Path.of(config.getGgufBasePath());
        }

        // Default to ~/.gollek/models
        return Path.of(System.getProperty("user.home"), ".gollek", "models");
    }

    private ProviderRegistry resolveProviderRegistry() {
        try {
            if (Arc.container() == null) {
                return null;
            }
            var instance = Arc.container().instance(ProviderRegistry.class);
            return instance.isAvailable() ? instance.get() : null;
        } catch (Exception ignored) {
            return null;
        }
    }
}
