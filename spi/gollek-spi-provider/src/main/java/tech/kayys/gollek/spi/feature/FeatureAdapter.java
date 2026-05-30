package tech.kayys.gollek.spi.feature;

/**
 * Marker SPI for out-of-core Gollek feature adapters.
 *
 * <p>Feature adapters describe capabilities that are wider than model
 * inference pipelines: training loops, quantizers, optimizers, backend
 * bridges, converters, dataset tools, evaluators, and other extension
 * surfaces. Runtime modules can discover implementations through
 * {@code META-INF/services/tech.kayys.gollek.spi.feature.FeatureAdapter} and
 * then route them to a domain-specific registry.</p>
 */
public interface FeatureAdapter {

    /**
     * Stable id used by manifests, CLI tooling, and registries.
     */
    String id();

    /**
     * Human- and machine-readable adapter metadata.
     */
    FeatureAdapterDescriptor descriptor();

    /**
     * Higher priority adapters are preferred when multiple adapters can handle
     * the same operation.
     */
    default int priority() {
        return 0;
    }
}
