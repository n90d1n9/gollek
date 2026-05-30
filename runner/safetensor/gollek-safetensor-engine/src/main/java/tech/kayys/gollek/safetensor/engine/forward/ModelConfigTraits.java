/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.util.Locale;

record ModelConfigTraits(
        ModelConfig source,
        String modelType,
        int hiddenSizePerLayerInput,
        int vocabSizePerLayerInput,
        boolean gemma4Text,
        boolean gemma3Text,
        boolean qwenText,
        boolean gemma4StylePerLayerInputs) {

    static final ModelConfigTraits EMPTY =
            new ModelConfigTraits(null, "", 0, 0, false, false, false, false);

    static ModelConfigTraits create(ModelConfig config) {
        return create(config, null);
    }

    static ModelConfigTraits create(ModelConfig config, ModelArchitecture arch) {
        String modelType = config.modelType() == null ? "" : config.modelType();
        String normalizedModelType = modelType.toLowerCase(Locale.ROOT);
        int hiddenSizePerLayerInput = config.hiddenSizePerLayerInput();
        int vocabSizePerLayerInput = config.vocabSizePerLayerInput();
        ModelRuntimeTraits runtimeTraits = arch == null
                ? ModelRuntimeTraits.fromConfig(config)
                : arch.runtimeTraits(config);
        return new ModelConfigTraits(
                config,
                modelType,
                hiddenSizePerLayerInput,
                vocabSizePerLayerInput,
                runtimeTraits.gemma4Text() || normalizedModelType.startsWith("gemma4"),
                runtimeTraits.gemma3Text() || normalizedModelType.startsWith("gemma3"),
                runtimeTraits.qwenText() || normalizedModelType.contains("qwen"),
                runtimeTraits.perLayerInputPath() || hiddenSizePerLayerInput > 0 || vocabSizePerLayerInput > 0);
    }

    boolean matches(ModelConfig config) {
        return source == config
                && modelType.equals(config.modelType() == null ? "" : config.modelType())
                && hiddenSizePerLayerInput == config.hiddenSizePerLayerInput()
                && vocabSizePerLayerInput == config.vocabSizePerLayerInput();
    }
}
