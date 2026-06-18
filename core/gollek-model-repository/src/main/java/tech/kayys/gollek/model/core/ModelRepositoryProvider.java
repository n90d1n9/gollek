package tech.kayys.gollek.model.core;

/**
 * Compatibility facade for the legacy repository-provider package.
 */
public interface ModelRepositoryProvider {

    String scheme();

    ModelRepository create(RepositoryContext context);
}
