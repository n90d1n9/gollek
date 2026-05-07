package tech.kayys.gollek.spi.provider;

import tech.kayys.gollek.spi.context.EngineContext;

import java.util.Optional;

/**
 * Context provided to providers during inference.
 */
public interface ProviderContext {

    /**
     * Get the context for this request.
     */
    Object getApiKeyContext();

    /**
     * Access to global engine services.
     */
    EngineContext getEngineContext();

    /**
     * Get a request-scoped attribute.
     */
    <T> Optional<T> getAttribute(String key, Class<T> type);

    /**
     * Set a request-scoped attribute.
     */
    void setAttribute(String key, Object value);
}
