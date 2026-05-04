/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.routing;

/**
 * Historical performance data for a specific execution provider.
 * This is used for predictive routing and anomaly detection.
 *
 * @param providerId      the ID of the provider
 * @param avgLatencyMs    average observed latency for this provider
 * @param successRate     the percentage of successful requests (0.0 to 1.0)
 * @param lastObservedUtc timestamp of the last data point
 */
public record RoutingHistory(
        String providerId,
        double avgLatencyMs,
        double successRate,
        long lastObservedUtc
) {
}
