/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardLinearProjection {
    private DirectForwardLinearProjection() {
    }

    static AccelTensor ffnDownLinear(DirectForwardRuntimeContext runtime,
                                     ModelConfigTraits traits,
                                     ModelConfig config,
                                     boolean decodeLogitsPhase,
                                     AccelTensor input,
                                     AccelTensor weight,
                                     AccelTensor bias,
                                     String profileKey,
                                     AccelTensor outputBuffer) {
        if (runtime.metalLinearEnabled()) {
            return linear(
                    runtime,
                    traits,
                    config,
                    decodeLogitsPhase,
                    input,
                    weight,
                    bias,
                    profileKey,
                    outputBuffer);
        }
        AccelTensor cached = tryCachedFfnDownHalfLinear(
                runtime,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                config,
                profileKey);
        if (cached != null) {
            return cached;
        }
        return linear(
                runtime,
                traits,
                config,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
    }

    static AccelTensor linear(DirectForwardRuntimeContext runtime,
                              ModelConfigTraits traits,
                              ModelConfig config,
                              boolean decodeLogitsPhase,
                              AccelTensor input,
                              AccelTensor weight,
                              AccelTensor bias,
                              String profileKey,
                              AccelTensor outputBuffer) {
        long t0 = profileKey != null ? System.nanoTime() : 0L;
        boolean metalLinearEnabled = runtime.metalLinearEnabled();
        if (weight.isQuantized()
                && weight.quantType() != AccelTensor.QuantType.BF16
                && weight.quantType() != AccelTensor.QuantType.F16) {
            AccelTensor dequantized = weight.dequantize();
            try {
                return linear(
                        runtime,
                        traits,
                        config,
                        decodeLogitsPhase,
                        input,
                        dequantized,
                        bias,
                        profileKey,
                        outputBuffer);
            } finally {
                if (dequantized != weight && !dequantized.isClosed()) {
                    dequantized.close();
                }
            }
        }

        AccelTensor metalHalf = DirectForwardMetalHalfLinear.tryLinear(
                runtime.log(),
                runtime.metalBinding(),
                runtime.capabilities(),
                traits,
                config,
                metalLinearEnabled,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                outputBuffer,
                profileKey);
        if (metalHalf != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalHalf;
        }

        AccelTensor metalFloat = DirectForwardMetalFloatLinear.tryLinear(
                runtime.log(), runtime.metalBinding(), metalLinearEnabled, input, weight, bias);
        if (metalFloat != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "metal_float_matmul");
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalFloat;
        }

        AccelTensor cachedLargeHalf = tryCachedLargeHalfLinear(
                runtime,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                profileKey);
        if (cachedLargeHalf != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return cachedLargeHalf;
        }

        AccelTensor mm = AccelOps.linear(input, weight);
        if (bias != null) {
            AccelTensor out = AccelOps.add(mm, bias);
            mm.close();
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear_with_bias");
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return out;
        }
        if (profileKey != null) {
            DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear");
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
        }
        return mm;
    }

    private static AccelTensor tryCachedFfnDownHalfLinear(DirectForwardRuntimeContext runtime,
                                                          boolean decodeLogitsPhase,
                                                          AccelTensor input,
                                                          AccelTensor weight,
                                                          AccelTensor bias,
                                                          ModelConfig config,
                                                          String profileKey) {
        AccelTensor dequantized = DirectForwardLinearCachePolicy.cachedFfnDownHalfWeight(
                input,
                weight,
                config,
                profileKey);
        if (dequantized == null) {
            return null;
        }
        return linear(
                runtime,
                ModelConfigTraits.EMPTY,
                null,
                decodeLogitsPhase,
                input,
                dequantized,
                bias,
                profileKey,
                null);
    }

    private static AccelTensor tryCachedLargeHalfLinear(DirectForwardRuntimeContext runtime,
                                                        boolean decodeLogitsPhase,
                                                        AccelTensor input,
                                                        AccelTensor weight,
                                                        AccelTensor bias,
                                                        String profileKey) {
        AccelTensor dequantized = DirectForwardLinearCachePolicy.cachedLogitsLargeHalfWeight(
                input,
                weight,
                profileKey);
        if (dequantized == null) {
            return null;
        }
        return linear(
                runtime,
                ModelConfigTraits.EMPTY,
                null,
                decodeLogitsPhase,
                input,
                dequantized,
                bias,
                profileKey,
                null);
    }
}
