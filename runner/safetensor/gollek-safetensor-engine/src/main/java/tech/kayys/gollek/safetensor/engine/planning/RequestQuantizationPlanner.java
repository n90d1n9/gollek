/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.util.Objects;

/**
 * Converts provider request quantization options into safetensor quantization policy.
 */
@ApplicationScoped
public class RequestQuantizationPlanner {
    private static final Logger log = Logger.getLogger(RequestQuantizationPlanner.class);

    QuantizationEngine.QuantStrategy resolve(ProviderRequest request) {
        Objects.requireNonNull(request, "request");
        String raw = request.getParameter("quantize_strategy", String.class).orElse("none");
        if (raw == null || raw.isBlank()) {
            return QuantizationEngine.QuantStrategy.NONE;
        }

        try {
            return QuantizationEngine.QuantStrategy.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            log.warnf("Unknown quantize_strategy '%s'; falling back to NONE", raw);
            return QuantizationEngine.QuantStrategy.NONE;
        }
    }
}
