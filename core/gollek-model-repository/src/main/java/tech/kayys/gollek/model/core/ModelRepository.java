package tech.kayys.gollek.model.core;

/**
 * Compatibility facade for the legacy model repository package.
 *
 * <p>The active repository contract now lives under {@code spi.model}, but a
 * large part of the codebase still imports the historical
 * {@code tech.kayys.gollek.model.core.ModelRepository} name.
 */
public interface ModelRepository extends tech.kayys.gollek.spi.model.ModelRepository {
}
