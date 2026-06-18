/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.sdk;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.nio.file.Files;
import java.nio.file.Path;

record LegacyDirectModelProfile(
        ModelConfig config,
        String modelType,
        ModelRuntimeTraits runtimeTraits) {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    LegacyDirectModelProfile {
        modelType = modelType == null ? "" : modelType;
        runtimeTraits = runtimeTraits == null ? ModelRuntimeTraits.EMPTY : runtimeTraits;
    }

    static LegacyDirectModelProfile load(Path modelPath, Logger log) {
        if (modelPath == null) {
            return unresolved();
        }
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir == null) {
                return unresolved();
            }
            ModelConfig config = ModelConfig.fromDirectory(configDir, OBJECT_MAPPER);
            return new LegacyDirectModelProfile(
                    config,
                    config.modelType(),
                    ModelRuntimeTraitsResolver.resolve(config));
        } catch (Exception e) {
            if (log != null) {
                log.debugf("Unable to read model profile for legacy direct prompt shaping: %s", e.getMessage());
            }
            return unresolved();
        }
    }

    static LegacyDirectModelProfile unresolved() {
        return new LegacyDirectModelProfile(null, "", ModelRuntimeTraits.EMPTY);
    }

    boolean gemma4Text() {
        return runtimeTraits.gemma4Text();
    }

    boolean gemma3Text() {
        return runtimeTraits.gemma3Text();
    }

    boolean hasKnownPromptTemplate() {
        if (runtimeTraits.gemma4Text() || runtimeTraits.gemma3Text() || runtimeTraits.qwenText()) {
            return true;
        }
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String normalized = modelType.trim().toLowerCase(java.util.Locale.ROOT)
                                     .replace("-", "").replace("_", "");
        return normalized.startsWith("llama3") ||
               normalized.startsWith("mistral") ||
               normalized.startsWith("mixtral") ||
               normalized.startsWith("phi") ||
               normalized.startsWith("gemma") ||
               normalized.startsWith("qwen") ||
               normalized.equals("chatml");
    }
}
