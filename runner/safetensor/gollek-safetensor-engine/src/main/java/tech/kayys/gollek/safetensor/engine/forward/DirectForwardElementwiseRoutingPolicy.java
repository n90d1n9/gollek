/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardElementwiseRoutingPolicy(DirectForwardElementwiseOptions options) {

    DirectForwardElementwiseRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardElementwiseRoutingPolicy from(DirectForwardElementwiseOptions options) {
        return new DirectForwardElementwiseRoutingPolicy(options);
    }

    boolean canUseMetalElementwise(
            ModelConfigTraits traits,
            int seqLen,
            boolean forceCpuForward,
            boolean canUseMetal,
            boolean nativeElementwiseKernelsAvailable,
            boolean nativeElementwiseFallbackAvailable) {
        if (forceCpuForward) {
            return false;
        }
        boolean gemma4 = traits.gemma4Text();
        if (gemma4) {
            if (options.disableMetalGemma4Elementwise()) {
                return false;
            }
            if (!canUseMetal || !nativeElementwiseKernelsAvailable) {
                return false;
            }
        } else if (!canUseMetal && !nativeElementwiseFallbackAvailable) {
            return false;
        }
        int defaultMinSeq = gemma4 ? 1 : 16;
        int minSeq = options.metalElementwiseMinSeq() >= 0
                ? options.metalElementwiseMinSeq()
                : defaultMinSeq;
        if (seqLen < minSeq) {
            return false;
        }
        if (!gemma4) {
            return true;
        }
        return options.enableGemma4MetalElementwise() || nativeElementwiseKernelsAvailable;
    }

    boolean canUseMetalLayerScalarScale(
            boolean useMetalElementwise,
            int seqLen,
            boolean metalBindingAvailable,
            boolean metalNativeScaleKernelAvailable) {
        return useMetalElementwise
                && metalBindingAvailable
                && metalNativeScaleKernelAvailable
                && (seqLen >= options.metalLayerScalarMinSeq()
                || (seqLen == 1 && options.enableMetalLayerScalarDecode()));
    }

    boolean shouldUseMetalPostFfnNorm(ModelConfigTraits traits) {
        if (options.disableMetalPostFfnNorm()) {
            return false;
        }
        if (options.enableMetalPostFfnNorm() != null) {
            return options.enableMetalPostFfnNorm();
        }
        return isGemma4FfnPolicyTarget(traits);
    }

    boolean shouldBuildPerLayerInputs(ModelConfigTraits traits, int hiddenSizePerLayerInput) {
        if (hiddenSizePerLayerInput <= 0) {
            return false;
        }
        return !isGemma4PerLayerInputDisabled(traits);
    }

    boolean shouldApplyLayerScalar(ModelConfigTraits traits) {
        if (!traits.gemma4Text()) {
            return true;
        }
        return !options.disableGemma4LayerScalar();
    }

    static boolean isGemma4FfnPolicyTarget(ModelConfigTraits traits) {
        return traits.gemma4Text() || traits.gemma4StylePerLayerInputs();
    }

    private boolean isGemma4PerLayerInputDisabled(ModelConfigTraits traits) {
        return traits.gemma4Text() && options.disableGemma4PerLayerInput();
    }
}
