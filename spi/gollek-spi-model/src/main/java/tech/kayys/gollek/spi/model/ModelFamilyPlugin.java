package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.plugin.PluginContext;

import java.util.List;

/**
 * Plugin contract for detachable Hugging Face style model families.
 *
 * <p>Model families are intentionally separate from runner plugins. A family
 * advertises config/tokenizer/weight-layout knowledge; a runner decides whether
 * it can execute that family for a specific artifact format.</p>
 */
public interface ModelFamilyPlugin extends GollekPlugin {

    ModelFamilyDescriptor descriptor();

    /**
     * Architecture adapters that are safe for Gollek's direct architecture
     * registry to resolve. Encoder-only or multimodal-only families can return
     * an empty list until their runner path is implemented.
     */
    default List<ModelArchitecture> architectureAdapters() {
        return List.of();
    }

    /**
     * Tokenizer profiles this family can use. Tokenizer runtimes choose a
     * matching profile only when its required files are present.
     */
    default List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of();
    }

    /**
     * Derived support status for diagnostics and runtime policy gates.
     */
    default ModelFamilySupportReport supportReport() {
        return ModelFamilySupportReport.from(this);
    }

    @Override
    default String id() {
        return "model-family/" + descriptor().id();
    }

    @Override
    default String version() {
        return descriptor().metadata().getOrDefault("version", "0.1.0-SNAPSHOT");
    }

    @Override
    default int order() {
        return 40;
    }

    @Override
    default void initialize(PluginContext context) {
        ModelFamilyPluginRegistry.global().register(this);
    }

    @Override
    default void stop() {
        ModelFamilyPluginRegistry.global().unregister(id());
    }

    @Override
    default void shutdown() {
        ModelFamilyPluginRegistry.global().unregister(id());
    }
}
