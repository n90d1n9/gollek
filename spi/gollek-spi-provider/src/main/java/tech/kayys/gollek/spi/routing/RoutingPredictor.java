/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.routing;

import tech.kayys.gollek.spi.provider.RoutingContext;

/**
 * Predictive engine that provides latency forecasts and reliability scores
 * based on {@link RoutingHistory} and other environmental factors.
 */
public interface RoutingPredictor {

    /**
     * Predict the latency for a given provider and request context.
     *
     * @param providerId the ID of the provider to predict for
     * @param ctx        the routing context for the prediction
     * @return predicted latency in milliseconds
     */
    double predictLatency(String providerId, RoutingContext ctx);

    /**
     * Provide a reliability score (0.0 to 1.0) based on historical success rates.
     *
     * @param providerId the ID of the provider to score
     * @return reliability score (1.0 is most reliable)
     */
    double predictReliability(String providerId);
}
