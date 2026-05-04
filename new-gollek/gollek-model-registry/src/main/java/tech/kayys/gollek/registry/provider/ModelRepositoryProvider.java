package tech.kayys.gollek.registry.provider;

import tech.kayys.gollek.model.core.RepositoryContext;
import tech.kayys.gollek.model.core.ModelRepository;

/**
 * Provider interface for model repository implementations.
 */
public interface ModelRepositoryProvider {

    String scheme(); // hf, local, s3, etc

    ModelRepository create(RepositoryContext context);
}
