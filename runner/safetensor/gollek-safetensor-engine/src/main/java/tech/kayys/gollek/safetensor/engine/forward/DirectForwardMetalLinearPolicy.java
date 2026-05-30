/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

final class DirectForwardMetalLinearPolicy {
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY = "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";
    private static final boolean EXPERIMENTAL_METAL_LINEAR_ENABLED =
            resolveExperimentalMetalLinearEnabled();
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final boolean DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_ENABLED =
            Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY);
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_VALUE =
            System.getProperty(EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY);
    private static final Boolean EXPERIMENTAL_METAL_BF16_LINEAR_EXPLICIT =
            parseOptionalBoolean(EXPERIMENTAL_METAL_BF16_LINEAR_VALUE);
    private static final String PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.prefer_native_metal_bf16_linear";
    private static final boolean PREFER_NATIVE_METAL_BF16_LINEAR_ENABLED =
            Boolean.getBoolean(PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY);
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_VALUE =
            System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    private static final Boolean ENABLE_GEMMA4_METAL_BF16_LINEAR_EXPLICIT =
            parseOptionalBoolean(ENABLE_GEMMA4_METAL_BF16_LINEAR_VALUE);
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final boolean DISABLE_GEMMA4_METAL_BF16_LINEAR_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LINEAR_VALUE =
            System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY);
    private static final Boolean ENABLE_GEMMA4_BF16_TO_F16_LINEAR_EXPLICIT =
            parseOptionalBoolean(ENABLE_GEMMA4_BF16_TO_F16_LINEAR_VALUE);
    private static final String DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_linear";
    private static final boolean DISABLE_GEMMA4_BF16_TO_F16_LINEAR_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY);
    private static final String ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_ffn_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_VALUE =
            System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY);
    private static final Boolean ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_EXPLICIT =
            parseOptionalBoolean(ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_VALUE);
    private static final String DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_ffn_linear";
    private static final boolean DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_PROPERTY);
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_logits_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_VALUE =
            System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY);
    private static final Boolean ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_EXPLICIT =
            parseOptionalBoolean(ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_VALUE);
    private static final String DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_logits_linear";
    private static final boolean DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_PROPERTY);

    private DirectForwardMetalLinearPolicy() {
    }

    static boolean experimentalMetalLinearEnabled() {
        return EXPERIMENTAL_METAL_LINEAR_ENABLED;
    }

    static boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        if (traits.gemma4Text()) {
            if (DISABLE_GEMMA4_METAL_BF16_LINEAR_ENABLED) {
                return false;
            }
            if (ENABLE_GEMMA4_METAL_BF16_LINEAR_EXPLICIT != null) {
                return ENABLE_GEMMA4_METAL_BF16_LINEAR_EXPLICIT;
            }
            return true;
        }
        return allowGenericMetalBf16Linear();
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return shouldUseNativeMetalBf16Linear(
                weight,
                traits,
                profileKey,
                allowGemma4Bf16ToF16Linear(traits, profileKey));
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        return preferNativeMetalBf16Linear(traits, profileKey, allowBf16ToF16)
                && allowMetalBf16Linear(traits)
                && weight != null
                && weight.quantType() == AccelTensor.QuantType.BF16;
    }

    static boolean canUseMetalHalfLinearCandidate(
            boolean canUseExperimentalMetalLinear,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseExperimentalMetalLinear) {
            return false;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (weight == null) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.gemma4Text()
                && !shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                && !allowGemma4Bf16ToF16Linear(traits, profileKey)) {
            return false;
        }
        if (quantType != AccelTensor.QuantType.F16
                && (quantType != AccelTensor.QuantType.BF16 || !allowMetalBf16Linear(traits))) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        int inputRank = input.rank();
        if (inputRank < 2 || weight.rank() != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, input.size(-1));
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputRank - 2; i++) {
            batchProduct *= input.size(i);
        }
        return batchProduct == 1L;
    }

    static boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfigTraits traits, String profileKey) {
        if (weight == null || weight.rank() != 2 || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.gemma4Text()) {
            return shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                    || allowGemma4Bf16ToF16Linear(traits, profileKey);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(traits));
    }

    static boolean allowGemma4Bf16ToF16Linear(ModelConfigTraits traits, String profileKey) {
        if (!traits.gemma4Text() || DISABLE_GEMMA4_BF16_TO_F16_LINEAR_ENABLED) {
            return false;
        }
        if ("logits".equals(profileKey) && DISABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_ENABLED) {
            return false;
        }
        if (ENABLE_GEMMA4_BF16_TO_F16_LINEAR_EXPLICIT != null) {
            return ENABLE_GEMMA4_BF16_TO_F16_LINEAR_EXPLICIT;
        }
        if ("logits".equals(profileKey)) {
            if (ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_EXPLICIT != null) {
                return ENABLE_GEMMA4_BF16_TO_F16_LOGITS_LINEAR_EXPLICIT;
            }
            return false;
        }
        if (!isGemma4FfnProjection(profileKey)
                || DISABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_ENABLED) {
            return false;
        }
        if (ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_EXPLICIT != null) {
            return ENABLE_GEMMA4_BF16_TO_F16_FFN_LINEAR_EXPLICIT;
        }
        return false;
    }

    static boolean allowGemma4Bf16ToF16LinearForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        // Keep prompt prefill on native BF16. Cached F16 conversion is useful
        // for repeated decode matvecs, but paying it before first token hurts TTFT.
        if (rows != 1L || !allowGemma4Bf16ToF16Linear(traits, profileKey)) {
            return false;
        }
        return !"logits".equals(profileKey) || decodeLogitsPhase;
    }

    private static boolean resolveExperimentalMetalLinearEnabled() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // Default ON for Apple Silicon safetensor text path to avoid accidental CPU-only runs.
        return true;
    }

    private static boolean allowGenericMetalBf16Linear() {
        if (DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_ENABLED) {
            return false;
        }
        if (EXPERIMENTAL_METAL_BF16_LINEAR_EXPLICIT != null) {
            return EXPERIMENTAL_METAL_BF16_LINEAR_EXPLICIT;
        }
        return false;
    }

    private static boolean preferNativeMetalBf16Linear(
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        if (traits.gemma4Text()) {
            if (allowBf16ToF16) {
                return false;
            }
            if (ENABLE_GEMMA4_METAL_BF16_LINEAR_EXPLICIT != null) {
                return ENABLE_GEMMA4_METAL_BF16_LINEAR_EXPLICIT && allowMetalBf16Linear(traits);
            }
            return allowMetalBf16Linear(traits);
        }
        return PREFER_NATIVE_METAL_BF16_LINEAR_ENABLED;
    }

    private static boolean isGemma4FfnProjection(String profileKey) {
        return profileKey != null && profileKey.startsWith("ffn_");
    }
}
