/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardNativeBf16MatvecPolicy.describeNativeBf16PairMatvecPath;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.addBiasIfNeeded;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isMultiRowLinearInput;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalHalfLinearPair {
    private static final String PATH = "gate-up-pair";
    private static final String PROFILE_KEY = "ffn_gate_up_pair";

    private DirectForwardMetalHalfLinearPair() {
    }

    static Result tryPair(Logger log,
                          MetalBinding metalBinding,
                          DirectForwardMetalCapabilities capabilities,
                          ModelConfigTraits traits,
                          ModelConfig config,
                          boolean metalLinearEnabled,
                          boolean decodeLogitsPhase,
                          AccelTensor input,
                          AccelTensor firstWeight,
                          AccelTensor firstBias,
                          AccelTensor secondWeight,
                          AccelTensor secondBias,
                          AccelTensor firstOutputBuffer,
                          AccelTensor secondOutputBuffer) {
        if (!shouldUsePair(traits, input)) {
            trace("reject:disabled", config, input, firstWeight, secondWeight);
            return null;
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, firstWeight, traits, PROFILE_KEY)) {
            trace("reject:first_candidate_ineligible", config, input, firstWeight, secondWeight);
            return null;
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, secondWeight, traits, PROFILE_KEY)) {
            trace("reject:second_candidate_ineligible", config, input, firstWeight, secondWeight);
            return null;
        }
        if (!capabilities.supportsMatmulTransposedRightHalfPair()) {
            trace("reject:pair_symbol_unavailable", config, input, firstWeight, secondWeight);
            return null;
        }
        if (firstWeight.size(0) != secondWeight.size(0)
                || firstWeight.size(1) != secondWeight.size(1)) {
            trace("reject:shape_mismatch", config, input, firstWeight, secondWeight);
            return null;
        }

        long k = input.size(-1);
        long rows = input.numel() / Math.max(1L, k);
        boolean allowBf16ToF16Weights = DirectForwardMetalLinearPolicy.allowGemma4Bf16ToF16LinearForRows(
                rows,
                traits,
                PROFILE_KEY,
                decodeLogitsPhase);
        boolean nativeBf16Weights = DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                firstWeight,
                traits,
                PROFILE_KEY,
                allowBf16ToF16Weights)
                && DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                secondWeight,
                traits,
                PROFILE_KEY,
                allowBf16ToF16Weights);
        AccelTensor metalFirstWeight = toMetalHalfWeight(
                firstWeight,
                traits,
                nativeBf16Weights,
                allowBf16ToF16Weights);
        AccelTensor metalSecondWeight = toMetalHalfWeight(
                secondWeight,
                traits,
                nativeBf16Weights,
                allowBf16ToF16Weights);
        if (metalFirstWeight == null || metalSecondWeight == null
                || metalFirstWeight.quantType() != metalSecondWeight.quantType()
                || (metalFirstWeight.quantType() != AccelTensor.QuantType.F16
                && metalFirstWeight.quantType() != AccelTensor.QuantType.BF16)) {
            trace("reject:weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, firstWeight, secondWeight);
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long outputDim = metalFirstWeight.size(0);
        long[] outputShape = input.shapeWithLastDim(outputDim);
        AccelTensor first = reusableOutputTensor(firstOutputBuffer, outputShape);
        AccelTensor second = reusableOutputTensor(secondOutputBuffer, outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(outputDim);
            int rc = -2;
            String executionPath = "metal_pair_matmul";
            if (m == 1
                    && nativeBf16Weights
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvecPair(traits, config, n)
                    && capabilities.supportsMatvecTransposedRightBf16Pair()) {
                rc = metalBinding.matvecTransposedRightBf16Pair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = describeNativeBf16PairMatvecPath(kk, n);
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weights
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvecPair(traits, config, n)
                    && capabilities.supportsMatvecTransposedRightHalfPair()) {
                rc = metalBinding.matvecTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "matvec";
                }
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPair failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, executionPath);
            DirectInferenceProfiler.recordLinearNanos(PROFILE_KEY, System.nanoTime() - t0);
            trace("accept:" + executionPath + ":native_bf16=" + nativeBf16Weights,
                    config, input, firstWeight, secondWeight);
            return new Result(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(),
                    config, input, firstWeight, secondWeight);
            log.debugf("Falling back from Metal half linear pair to separate linears: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static boolean shouldUsePair(ModelConfigTraits traits, AccelTensor input) {
        return DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfLinearPair(
                traits,
                isMultiRowLinearInput(input),
                DirectForwardFfnFastPathPolicy.allowGemma4FusedHalfFfn());
    }

    private static AccelTensor toMetalHalfWeight(AccelTensor weight,
                                                 ModelConfigTraits traits,
                                                 boolean nativeBf16,
                                                 boolean allowBf16ToF16) {
        return DirectForwardLinearCachePolicy.toMetalHalfWeight(
                weight,
                nativeBf16,
                traits.gemma4Text(),
                allowBf16ToF16,
                DirectForwardMetalLinearPolicy.allowMetalBf16Linear(traits));
    }

    private static void trace(String decision,
                              ModelConfig config,
                              AccelTensor input,
                              AccelTensor firstWeight,
                              AccelTensor secondWeight) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, firstWeight, secondWeight, null);
    }

    record Result(AccelTensor first, AccelTensor second) {
    }
}
