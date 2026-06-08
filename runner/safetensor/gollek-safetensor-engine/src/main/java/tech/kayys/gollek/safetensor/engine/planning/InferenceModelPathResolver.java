/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Resolves request model identifiers into local safetensor model paths.
 */
@ApplicationScoped
public class InferenceModelPathResolver {
    private static final Logger log = Logger.getLogger(InferenceModelPathResolver.class);

    Path resolve(ProviderRequest request, SafetensorProviderConfig config) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(config, "config");

        java.util.Optional<String> fromParam = request.getParameter("model_path", String.class);
        if (fromParam.isPresent() && !fromParam.get().isBlank()) {
            log.debugf("InferenceModelPathResolver: using model_path from parameters: %s", fromParam.get());
            return resolveModelPath(fromParam.get(), config);
        }

        Object fromMeta = request.getMetadata().get("model_path");
        if (fromMeta instanceof String s && !s.isBlank()) {
            log.debugf("InferenceModelPathResolver: using model_path from metadata: %s", s);
            return resolveModelPath(s, config);
        }

        log.debugf("InferenceModelPathResolver: no model_path override, resolving from model ID: %s", request.getModel());
        return resolveModelPath(request.getModel(), config);
    }

    private Path resolveModelPath(String modelId, SafetensorProviderConfig config) {
        if (modelId == null || modelId.isBlank()) {
            throw new ProviderException("Model ID is null or blank");
        }

        Path asPath = Path.of(modelId);
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        Path resolved = Path.of(config.basePath(), modelId);
        if (Files.exists(resolved)) {
            log.debugf("Resolved model '%s' to path: %s", modelId, resolved);
            return resolved;
        }

        log.warnf("Model path not found at %s, using as-is: %s", resolved, modelId);
        return asPath;
    }
}
