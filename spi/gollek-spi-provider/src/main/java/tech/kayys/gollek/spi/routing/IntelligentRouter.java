/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.routing;

import tech.kayys.gollek.spi.provider.RoutingContext;

/**
 * Intelligent Router Engine SPI.
 *
 * <p>Decides the best runtime/provider to execute a request based on
 * available capabilities, live costs, latencies, and routing preferences.</p>
 */
public interface IntelligentRouter {

    /**
     * Determine the optimal routing decision.
     *
     * @param context the context describing the request and its constraints
     * @return the routing decision containing the selected provider
     */
    RoutingDecision route(RoutingContext context);
}
