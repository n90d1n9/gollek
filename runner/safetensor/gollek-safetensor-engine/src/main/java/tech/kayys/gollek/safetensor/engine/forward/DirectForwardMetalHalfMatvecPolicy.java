/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalHalfMatvecPolicy {
    private static final String ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_linear_pair";
    private static final String ENABLE_METAL_HALF_LINEAR_PAIR_VALUE =
            System.getProperty(ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY);
    private static final Boolean ENABLE_METAL_HALF_LINEAR_PAIR_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_HALF_LINEAR_PAIR_VALUE);
    private static final String DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_linear_pair";
    private static final boolean DISABLE_METAL_HALF_LINEAR_PAIR_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY);
    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_VALUE =
            System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY);
    private static final Boolean AUTO_METAL_HALF_MATVEC_EXPLICIT =
            parseOptionalBoolean(AUTO_METAL_HALF_MATVEC_VALUE);
    private static final boolean AUTO_METAL_HALF_MATVEC_ENABLED =
            resolveAutoMetalHalfMatvecEnabled();
    private static final String ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec_pair";
    private static final String DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec_pair";
    private static final boolean DISABLE_METAL_HALF_MATVEC_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY);
    private static final boolean DISABLE_METAL_HALF_MATVEC_PAIR_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY);
    private static final String ENABLE_METAL_HALF_MATVEC_VALUE =
            System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY);
    private static final Boolean ENABLE_METAL_HALF_MATVEC_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_HALF_MATVEC_VALUE);
    private static final String ENABLE_METAL_HALF_MATVEC_PAIR_VALUE =
            System.getProperty(ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY);
    private static final Boolean ENABLE_METAL_HALF_MATVEC_PAIR_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_HALF_MATVEC_PAIR_VALUE);
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 24576;
    private static final int METAL_HALF_MATVEC_MAX_OUTPUT =
            Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY, DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT);
    private static final String ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_logits_mps_matvec";
    private static final String ENABLE_METAL_LOGITS_MPS_MATVEC_VALUE =
            System.getProperty(ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY);
    private static final Boolean ENABLE_METAL_LOGITS_MPS_MATVEC_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_LOGITS_MPS_MATVEC_VALUE);
    private static final String DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_logits_mps_matvec";
    private static final boolean DISABLE_METAL_LOGITS_MPS_MATVEC_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY);
    private static final String METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_min_output";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT = 65536;
    private static final int METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT =
            Integer.getInteger(METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY,
                    DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT);
    private static final String METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_max_input";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT = 4096;
    private static final int METAL_LOGITS_MPS_MATVEC_MAX_INPUT =
            Integer.getInteger(METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY,
                    DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT);
    private static final String GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.gemma4_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final int GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT =
            Integer.getInteger(GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                    DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT);
    private static final String ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_gemma3_logits_metal_half_matvec";
    private static final String ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_VALUE =
            System.getProperty(ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY);
    private static final Boolean ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_EXPLICIT =
            parseOptionalBoolean(ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_VALUE);
    private static final String DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_gemma3_logits_metal_half_matvec";
    private static final boolean DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_PROPERTY);
    private static final String GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.gemma3_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final int GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT =
            Integer.getInteger(GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                    DEFAULT_GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT);
    private static final String QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.qwen_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final int QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT =
            Integer.getInteger(QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                    DEFAULT_QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT);
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_VALUE =
            System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY);
    private static final Boolean ENABLE_METAL_TRANSPOSED_HALF_MATVEC_EXPLICIT =
            parseOptionalBoolean(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_VALUE);
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec_all";
    private static final boolean ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_ENABLED =
            Boolean.getBoolean(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY);
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final boolean DISABLE_METAL_TRANSPOSED_HALF_MATVEC_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY);
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final int METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT =
            Integer.getInteger(METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                    DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT);
    private static final boolean METAL_HALF_LINEAR_PAIR_ENABLED =
            resolveMetalHalfLinearPairEnabled();

    private DirectForwardMetalHalfMatvecPolicy() {
    }

    static boolean shouldUseMetalHalfLinearPair(
            ModelConfigTraits traits,
            boolean multiRowLinearInput,
            boolean allowGemma4FusedHalfFfn) {
        if (!METAL_HALF_LINEAR_PAIR_ENABLED) {
            return false;
        }
        if (ENABLE_METAL_HALF_LINEAR_PAIR_EXPLICIT != null) {
            return ENABLE_METAL_HALF_LINEAR_PAIR_EXPLICIT;
        }
        if (traits.gemma4Text() || traits.gemma4StylePerLayerInputs()) {
            return multiRowLinearInput || !allowGemma4FusedHalfFfn;
        }
        return true;
    }

    static boolean shouldUseMetalHalfMatvec(
            ModelConfigTraits traits,
            ModelConfig config,
            int outputDim,
            String profileKey) {
        if (DISABLE_METAL_HALF_MATVEC_ENABLED) {
            return false;
        }
        int maxOutput = metalHalfMatvecMaxOutput(traits, profileKey);
        if (shouldUseGemma3LogitsMetalHalfMatvec(traits, outputDim, profileKey, maxOutput)) {
            return true;
        }
        if (ENABLE_METAL_HALF_MATVEC_EXPLICIT != null) {
            return ENABLE_METAL_HALF_MATVEC_EXPLICIT && maxOutput > 0 && outputDim <= maxOutput;
        }
        return shouldAutoUseMetalHalfMatvec(traits, config, outputDim, maxOutput);
    }

    static boolean shouldUseMetalHalfMatvecPair(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        if (DISABLE_METAL_HALF_MATVEC_PAIR_ENABLED) {
            return false;
        }
        if (ENABLE_METAL_HALF_MATVEC_PAIR_EXPLICIT != null) {
            return ENABLE_METAL_HALF_MATVEC_PAIR_EXPLICIT && outputDim <= metalHalfMatvecMaxOutput();
        }
        return shouldAutoUseMetalHalfMatvec(traits, config, outputDim);
    }

    static boolean shouldUseMetalLogitsMpsMatvec(
            ModelConfigTraits traits,
            int outputDim,
            int inputDim,
            String profileKey) {
        if (!"logits".equals(profileKey) || DISABLE_METAL_LOGITS_MPS_MATVEC_ENABLED) {
            return false;
        }
        if (!Boolean.TRUE.equals(ENABLE_METAL_LOGITS_MPS_MATVEC_EXPLICIT)) {
            return false;
        }
        if (traits.gemma4Text()) {
            return false;
        }
        return outputDim >= METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT
                && (METAL_LOGITS_MPS_MATVEC_MAX_INPUT <= 0
                || inputDim <= METAL_LOGITS_MPS_MATVEC_MAX_INPUT);
    }

    static boolean shouldUseMetalTransposedHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey) {
        if (DISABLE_METAL_TRANSPOSED_HALF_MATVEC_ENABLED) {
            return false;
        }
        int maxOutput = METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT;
        if (maxOutput <= 0 || outputDim > maxOutput) {
            return false;
        }
        boolean explicitEnabled = Boolean.TRUE.equals(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_EXPLICIT);
        boolean allowAllExperimentalProjectors = ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_ENABLED;
        boolean logitsProjection = "logits".equals(profileKey);
        if (!logitsProjection) {
            // The transposed-weight matvec kernel is still experimental for
            // hidden projections; keep global opt-in safe for logits unless a
            // caller explicitly asks to experiment across all projectors.
            return explicitEnabled && allowAllExperimentalProjectors;
        }
        if (!traits.gemma4Text() && !allowAllExperimentalProjectors) {
            return false;
        }
        if (ENABLE_METAL_TRANSPOSED_HALF_MATVEC_EXPLICIT != null) {
            return explicitEnabled;
        }
        return traits.gemma4Text();
    }

    private static boolean resolveMetalHalfLinearPairEnabled() {
        if (DISABLE_METAL_HALF_LINEAR_PAIR_ENABLED) {
            return false;
        }
        if (ENABLE_METAL_HALF_LINEAR_PAIR_EXPLICIT != null) {
            return ENABLE_METAL_HALF_LINEAR_PAIR_EXPLICIT;
        }
        return true;
    }

    private static boolean resolveAutoMetalHalfMatvecEnabled() {
        if (AUTO_METAL_HALF_MATVEC_EXPLICIT != null) {
            return AUTO_METAL_HALF_MATVEC_EXPLICIT;
        }
        return true;
    }

    private static boolean shouldAutoUseMetalHalfMatvec(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        return shouldAutoUseMetalHalfMatvec(traits, config, outputDim, metalHalfMatvecMaxOutput());
    }

    private static boolean shouldAutoUseMetalHalfMatvec(
            ModelConfigTraits traits,
            ModelConfig config,
            int outputDim,
            int maxOutput) {
        return AUTO_METAL_HALF_MATVEC_ENABLED
                && maxOutput > 0
                && outputDim <= maxOutput
                && isMetalHalfMatvecAutoCandidate(traits, config);
    }

    private static boolean isMetalHalfMatvecAutoCandidate(ModelConfigTraits traits, ModelConfig config) {
        if (traits.modelType().isBlank()) {
            return false;
        }
        if (traits.gemma4Text() || traits.gemma4StylePerLayerInputs()) {
            return true;
        }
        if (traits.qwenText()) {
            return config.numHiddenLayers() >= 20
                    && config.hiddenSize() <= 2048
                    && config.intermediateSize() >= 2048;
        }
        return config.numHiddenLayers() >= 30
                && config.intermediateSize() >= 4096
                && config.hiddenSize() <= 4096;
    }

    private static int metalHalfMatvecMaxOutput() {
        return METAL_HALF_MATVEC_MAX_OUTPUT;
    }

    private static int metalHalfMatvecMaxOutput(ModelConfigTraits traits, String profileKey) {
        if ("logits".equals(profileKey) && traits.gemma4Text()) {
            return GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT;
        }
        if ("logits".equals(profileKey) && traits.gemma3Text()) {
            return GEMMA3_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT;
        }
        if ("logits".equals(profileKey) && traits.qwenText()) {
            return QWEN_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT;
        }
        return metalHalfMatvecMaxOutput();
    }

    private static boolean shouldUseGemma3LogitsMetalHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey,
            int maxOutput) {
        if (!"logits".equals(profileKey)
                || !traits.gemma3Text()
                || DISABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_ENABLED) {
            return false;
        }
        if (maxOutput <= 0 || outputDim > maxOutput) {
            return false;
        }
        // M4 profiles currently prefer the default MPS matmul logits path for
        // FunctionGemma; keep this branch explicit so experiments cannot drift
        // into the default decode policy.
        return Boolean.TRUE.equals(ENABLE_GEMMA3_LOGITS_METAL_HALF_MATVEC_EXPLICIT);
    }
}
