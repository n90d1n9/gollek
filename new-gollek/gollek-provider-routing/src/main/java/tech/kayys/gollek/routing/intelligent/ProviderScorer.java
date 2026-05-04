/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.routing.intelligent;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingPredictor;
import tech.kayys.gollek.spi.runtime.CostProfile;
import tech.kayys.gollek.spi.runtime.ExecutionProvider;
import tech.kayys.gollek.spi.runtime.RoutingPreference;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;

/**
 * Executes the scoring algorithm across multiple dimensions
 * (Cost, Latency, Quality) according to the weights supplied
 * by the requester's RoutingPreference.
 */
@ApplicationScoped
public class ProviderScorer {

    private final Instance<RoutingPredictor> predictors;

    @Inject
    public ProviderScorer(Instance<RoutingPredictor> predictors) {
        this.predictors = predictors;
    }

    /**
     * Calculates the composite score for a provider.
     * Lower is better (representing relative "cost" abstractly).
     */
    public double score(ExecutionProvider provider, RoutingContext ctx) {
        CostProfile cost = provider.costProfile();
        RoutingPreference pref = ctx.preference();

        double latencyScore = normalizeLatency(cost.latencyMs());
        double costScore = normalizeCost(cost.costPer1KTokens());
        double qualityScore = normalizeQuality(cost.qualityScore());

        double loadPenalty = computeLoadPenalty(provider);
        double predictionAdjustment = computePredictionAdjustment(provider, ctx);
        
        // Locality bonus: prefer local providers or those in the same region
        double localityBonus = 0.0;
        if (pref.preferLocal()) {
            if (provider.isLocal()) {
                localityBonus -= 0.15; // Strong local bonus
            } else if (ctx.region().isPresent() && ctx.region().get().equals(provider.node().region())) {
                localityBonus -= 0.08; // Regional bonus
            }
        }

        return pref.latencyWeight() * latencyScore
             + pref.costWeight() * costScore
             + pref.qualityWeight() * qualityScore
             + loadPenalty
             + predictionAdjustment
             + (localityBonus + (provider.node().load() * 0.1)); // penalize heavily loaded nodes
    }

    private double computePredictionAdjustment(ExecutionProvider provider, RoutingContext ctx) {
        if (predictors.isUnsatisfied()) {
            return 0.0;
        }

        return predictors.stream()
                .mapToDouble(p -> {
                    double predictedLatency = p.predictLatency(provider.id(), ctx);
                    double reliability = p.predictReliability(provider.id());
                    
                    // Penalty for high predicted latency relative to static profile
                    double latencyDelta = Math.max(0.0, predictedLatency - provider.costProfile().latencyMs());
                    double latencyPenalty = (latencyDelta / 1000.0) * 0.5;
                    
                    // Penalty for low reliability
                    double reliabilityPenalty = (1.0 - reliability) * 0.3;
                    
                    return latencyPenalty + reliabilityPenalty;
                })
                .average()
                .orElse(0.0);
    }

    private double normalizeLatency(double latencyMs) {
        // Base 1000ms mapping. Above 1000ms score scales up fast.
        return Math.max(0.0, latencyMs) / 1000.0;
    }

    private double normalizeCost(double costPer1KTokens) {
        // Assume $10 as a "very expensive" baseline for normalization
        return Math.max(0.0, costPer1KTokens) / 10.0;
    }

    private double normalizeQuality(double qualityScore) {
        // quality is 0.0 to 1.0 where 1.0 is highest quality.
        // We want lower score for better quality.
        return 1.0 - Math.min(1.0, Math.max(0.0, qualityScore));
    }

    private double computeLoadPenalty(ExecutionProvider provider) {
        // TODO: integrate with scheduler load metrics
        return 0.0;
    }
}
