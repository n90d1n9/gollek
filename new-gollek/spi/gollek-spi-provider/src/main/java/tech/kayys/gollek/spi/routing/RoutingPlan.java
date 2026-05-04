/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.routing;

/**
 * Strategy for distributing an inference request across one or more execution providers.
 *
 * <p>Supports simple single-provider routing, multi-phase (Prefill/Decode) splitting,
 * and speculative decoding plans.</p>
 */
public sealed interface RoutingPlan
        permits RoutingPlan.Single, RoutingPlan.Split, RoutingPlan.Speculative {

    /**
     * A simple plan where a single provider handles all inference phases.
     */
    record Single(String providerId) implements RoutingPlan {
    }

    /**
     * A plan that splits the inference across two different providers.
     * Usually, prompt processing (prefill) is routed to high-throughput compute (GPU),
     * while token generation (decode) stays on efficient compute (CPU/NPU).
     */
    record Split(String prefillProviderId, String decodeProviderId) implements RoutingPlan {
    }

    /**
     * A plan for speculative decoding where a draft model predicts tokens
     * and a target model verifies them.
     */
    record Speculative(String draftProviderId, String targetProviderId) implements RoutingPlan {
    }
}
