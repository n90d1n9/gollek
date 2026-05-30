/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.allFinite;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isSingleRowLinearInput;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalMatvecGatedFfn {
    private static final String PATH = "matvec-gated-ffn";
    private static final String PROFILE_KEY = "ffn_matvec_gated";

    private DirectForwardMetalMatvecGatedFfn() {
    }

    static boolean shouldAttempt(AccelTensor input, ModelConfigTraits traits, FFNActivationType activationType) {
        if (!isSingleRowLinearInput(input)) {
            return false;
        }
        return switch (activationType) {
            case GELU -> DirectForwardFfnFastPathPolicy.shouldUseMetalGegluMatvecFfn(traits);
            case SILU -> DirectForwardFfnFastPathPolicy.shouldUseMetalSwigluMatvecFfn(traits);
            default -> false;
        };
    }

    static AccelTensor tryFfn(Logger log,
                              MetalBinding metalBinding,
                              DirectForwardMetalCapabilities capabilities,
                              ModelConfigTraits traits,
                              ModelConfig config,
                              boolean metalLinearEnabled,
                              boolean decodeLogitsPhase,
                              AccelTensor input,
                              FFNActivationType activationType,
                              AccelTensor gateW,
                              AccelTensor gateB,
                              AccelTensor upW,
                              AccelTensor upB,
                              AccelTensor downW,
                              AccelTensor downB,
                              AccelTensor outputBuffer) {
        boolean siluActivation = activationType == FFNActivationType.SILU;
        boolean geluActivation = activationType == FFNActivationType.GELU;
        if (geluActivation && !DirectForwardFfnFastPathPolicy.shouldUseMetalGegluMatvecFfn(traits)) {
            trace("reject:geglu_flag_disabled", config, input, gateW, upW, downW);
            return null;
        }
        if (siluActivation && !DirectForwardFfnFastPathPolicy.shouldUseMetalSwigluMatvecFfn(traits)) {
            trace("reject:swiglu_flag_disabled", config, input, gateW, upW, downW);
            return null;
        }
        if (!siluActivation && !geluActivation) {
            trace("reject:unsupported_activation:" + activationType, config, input, gateW, upW, downW);
            return null;
        }
        if (gateB != null || upB != null || downB != null) {
            trace("reject:bias_present", config, input, gateW, upW, downW);
            return null;
        }
        if (!metalLinearEnabled || metalBinding == null) {
            trace("reject:metal_unavailable", config, input, gateW, upW, downW);
            return null;
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, gateW, traits, PROFILE_KEY)
                || !DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, upW, traits, PROFILE_KEY)
                || !DirectForwardMetalLinearPolicy.canUseMetalHalfWeight(downW, traits, PROFILE_KEY)) {
            trace("reject:candidate_ineligible", config, input, gateW, upW, downW);
            return null;
        }

        if (gateW.rank() != 2
                || upW.rank() != 2
                || downW.rank() != 2
                || gateW.size(0) != upW.size(0)
                || gateW.size(1) != upW.size(1)
                || downW.size(1) != gateW.size(0)) {
            trace("reject:shape_mismatch", config, input, gateW, upW, downW);
            return null;
        }

        long inputDim = input.size(-1);
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows != 1L) {
            trace("reject:not_single_token_rows:" + rows, config, input, gateW, upW, downW);
            return null;
        }

        boolean allowBf16ToF16Weights = DirectForwardMetalLinearPolicy.allowGemma4Bf16ToF16LinearForRows(
                rows,
                traits,
                PROFILE_KEY,
                decodeLogitsPhase);
        boolean nativeBf16Weights = DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                gateW, traits, PROFILE_KEY, allowBf16ToF16Weights)
                && DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                upW, traits, PROFILE_KEY, allowBf16ToF16Weights)
                && DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                downW, traits, PROFILE_KEY, allowBf16ToF16Weights);
        if (nativeBf16Weights && siluActivation && !capabilities.supportsSwigluFfnMatvecBf16()) {
            trace("reject:swiglu_bf16_matvec_symbol_unavailable", config, input, gateW, upW, downW);
            return null;
        }
        if (nativeBf16Weights && geluActivation && !capabilities.supportsGegluFfnMatvecBf16()) {
            trace("reject:geglu_bf16_matvec_symbol_unavailable", config, input, gateW, upW, downW);
            return null;
        }
        if (!nativeBf16Weights && siluActivation && !capabilities.supportsSwigluFfnMatvecHalf()) {
            trace("reject:swiglu_matvec_symbol_unavailable", config, input, gateW, upW, downW);
            return null;
        }
        if (!nativeBf16Weights && geluActivation && !capabilities.supportsGegluFfnMatvecHalf()) {
            trace("reject:geglu_matvec_symbol_unavailable", config, input, gateW, upW, downW);
            return null;
        }

        AccelTensor metalGateW = toMetalHalfWeight(gateW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        AccelTensor metalUpW = toMetalHalfWeight(upW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        AccelTensor metalDownW = toMetalHalfWeight(downW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        if (metalGateW == null || metalUpW == null || metalDownW == null) {
            trace("reject:weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return null;
        }
        AccelTensor.QuantType weightType = metalGateW.quantType();
        if (metalUpW.quantType() != weightType
                || metalDownW.quantType() != weightType
                || (weightType != AccelTensor.QuantType.F16 && weightType != AccelTensor.QuantType.BF16)) {
            trace("reject:weight_type_mismatch:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long intermediateDim = metalGateW.size(0);
        long outputDim = metalDownW.size(0);
        long[] outputShape = input.shapeWithLastDim(outputDim);
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int rc;
            if (nativeBf16Weights && siluActivation) {
                rc = metalBinding.swigluFfnMatvecBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else if (nativeBf16Weights) {
                rc = metalBinding.gegluFfnMatvecBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else if (siluActivation) {
                rc = metalBinding.swigluFfnMatvecHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else {
                rc = metalBinding.gegluFfnMatvecHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matvec gated FFN failed with code " + rc);
            }
            if (DirectForwardFfnFastPathPolicy.shouldValidateMetalMatvecFfn(
                    DirectForwardFfnFastPathTrace.isEnabled()) && !allFinite(out)) {
                throw new IllegalStateException("Metal matvec gated FFN produced non-finite output");
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY,
                    (siluActivation ? "swiglu" : "geglu") + "_matvec"
                            + (nativeBf16Weights ? "_bf16" : "_f16"));
            DirectInferenceProfiler.recordLinearNanos(
                    nativeBf16Weights
                            ? (siluActivation ? "ffn_swiglu_matvec_bf16" : "ffn_geglu_matvec_bf16")
                            : (siluActivation ? "ffn_swiglu_matvec_metal" : "ffn_geglu_matvec_metal"),
                    System.nanoTime() - t0);
            trace("accept:" + (siluActivation ? "swiglu" : "geglu") + ":native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW, downW);
            log.debugf("Falling back from Metal matvec gated FFN: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
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
                              AccelTensor gateW,
                              AccelTensor upW,
                              AccelTensor downW) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, gateW, upW, downW);
    }
}
