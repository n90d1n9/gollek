/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

record DirectSafetensorRunProfile(
        ModelConfig config,
        String modelType,
        ModelRuntimeTraits runtimeTraits) {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    DirectSafetensorRunProfile {
        modelType = modelType == null ? "" : modelType;
        runtimeTraits = runtimeTraits == null ? ModelRuntimeTraits.EMPTY : runtimeTraits;
    }

    static DirectSafetensorRunProfile load(Path modelPath) {
        if (modelPath == null) {
            return unresolved();
        }
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir == null) {
                return unresolved();
            }
            ModelConfig config = ModelConfig.fromDirectory(configDir, OBJECT_MAPPER);
            return new DirectSafetensorRunProfile(
                    config,
                    config.modelType(),
                    ModelRuntimeTraits.fallbackFromConfig(config));
        } catch (Exception ignored) {
            return unresolved();
        }
    }

    static DirectSafetensorRunProfile unresolved() {
        return new DirectSafetensorRunProfile(null, "", ModelRuntimeTraits.EMPTY);
    }

    boolean gemma4Text() {
        return runtimeTraits.gemma4Text();
    }

    boolean gemma4Unified() {
        if (config == null) {
            return false;
        }
        String normalizedModelType = normalize(config.modelType());
        String normalizedArchitecture = normalize(config.primaryArchitecture());
        return normalizedModelType.equals("gemma4_unified")
                || normalizedArchitecture.equals("gemma4unifiedforconditionalgeneration")
                || normalizedArchitecture.equals("gemma4formultimodallm")
                || normalizedArchitecture.equals("gemma4forimagetexttotext");
    }

    boolean gemma3Text() {
        return runtimeTraits.gemma3Text();
    }

    boolean qwenText() {
        return runtimeTraits.qwenText();
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}
