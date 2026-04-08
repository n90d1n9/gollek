package tech.kayys.gollek.runtime.unified.impl;

import tech.kayys.gollek.spi.context.RequestContext;

/**
 * Shared context for the unified runtime.
 *
 * <p>Provides a central location for the static {@link RequestContext} instance 
 * used to identify requests originating from the unified distribution.
 *
 * @author Bhangun
 * @since 1.0.0
 */
public class UnifiedRuntimeContext {

    /**
     * The singleton {@link RequestContext} instance for this runtime.
     */
    public static final RequestContext INSTANCE = RequestContext.of("unified-runtime");

    private UnifiedRuntimeContext() {
        // Prevent instantiation
    }
}