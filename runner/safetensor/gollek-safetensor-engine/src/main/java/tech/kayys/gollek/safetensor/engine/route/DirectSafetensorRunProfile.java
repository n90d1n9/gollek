/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;

import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.loader.ModelConfigLoader;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Lightweight direct SafeTensor run profile derived before the full model payload is loaded.
 *
 * <p>The CLI uses this profile for routing, prompt formatting, and preflight guards, so
 * family-specific runtime policy should be applied here instead of falling back to broad
 * config-name heuristics when the family module is available.</p>
 */
public record DirectSafetensorRunProfile(
        ModelConfig config,
        String modelType,
        ModelRuntimeTraits runtimeTraits) {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public DirectSafetensorRunProfile {
        modelType = modelType == null ? "" : modelType;
        runtimeTraits = runtimeTraits == null ? ModelRuntimeTraits.EMPTY : runtimeTraits;
    }

    public static DirectSafetensorRunProfile load(Path modelPath) {
        if (modelPath == null) {
            return unresolved();
        }
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir == null) {
                return unresolved();
            }
            ModelConfig config = new ModelConfigLoader(OBJECT_MAPPER).loadFromDirectory(configDir);
            return new DirectSafetensorRunProfile(
                    config,
                    config.getModelType(),
                    runtimeTraits(config));
        } catch (Exception ignored) {
            return unresolved();
        }
    }

    public static DirectSafetensorRunProfile unresolved() {
        return new DirectSafetensorRunProfile(null, "", ModelRuntimeTraits.EMPTY);
    }

    public boolean gemma4Text() {
        return runtimeTraits.gemma4Text();
    }

    public boolean gemma4Unified() {
        if (config == null) {
            return false;
        }
        String normalizedModelType = normalize(config.getModelType());
        String normalizedArchitecture = normalize(config.getPrimaryArchitecture());
        return normalizedModelType.equals("gemma4_unified")
                || normalizedArchitecture.equals("gemma4unifiedforconditionalgeneration")
                || normalizedArchitecture.equals("gemma4formultimodallm")
                || normalizedArchitecture.equals("gemma4forimagetexttotext");
    }

    public boolean gemma3Text() {
        return runtimeTraits.gemma3Text();
    }

    public boolean qwenText() {
        return runtimeTraits.qwenText();
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }
}
