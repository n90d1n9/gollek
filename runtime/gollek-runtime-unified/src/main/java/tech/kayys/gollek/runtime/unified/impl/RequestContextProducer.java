package tech.kayys.gollek.runtime.unified.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.enterprise.inject.Produces;
import tech.kayys.gollek.spi.context.RequestContext;

/**
 * CDI producer for the unified {@link RequestContext}.
 *
 * <p>Exposes the unified runtime context to the application, allowing 
 * other components to inject the identity of the running distribution.
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
public class RequestContextProducer {

    /**
     * Produces the default {@link RequestContext} for the unified runtime.
     *
     * @return The singleton {@link RequestContext} instance.
     */
    @Produces
    @Default
    public RequestContext produceRequestContext() {
        return UnifiedRuntimeContext.INSTANCE;
    }
}