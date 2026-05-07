package tech.kayys.gollek.spi.model;
import tech.kayys.gollek.spi.spec.*;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.model.ModelFormat;

/**
 * Provider interface for model repositories.
 */
public interface ModelRepositoryProvider {

    /**
     * @return the scheme supported by this provider (e.g., "hf", "local", "s3")
     */
    String scheme();

    /**
     * Create a new instance of the model repository.
     * 
     * @param context the repository context
     * @return the model repository instance
     */
    ModelRepository create(RepositoryContext context);
}
