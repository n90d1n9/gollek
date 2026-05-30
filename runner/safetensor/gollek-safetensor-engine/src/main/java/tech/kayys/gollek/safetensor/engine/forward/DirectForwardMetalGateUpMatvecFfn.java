/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalGateUpMatvecFfn {
    private static final String PATH = "gate-up-gated-matvec";
    private static final String PROFILE_KEY = "ffn_gate_up_gated";

    private DirectForwardMetalGateUpMatvecFfn() {
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
                              AccelTensor combinedBuffer) {
        boolean siluActivation = activationType == FFNActivationType.SILU;
        boolean geluActivation = activationType == FFNActivationType.GELU;
        if (!DirectForwardFfnFastPathPolicy.shouldUseMetalGateUpMatvecFfn()) {
            trace("reject:flag_disabled", config, input, gateW, upW);
            return null;
        }
        if (!siluActivation && !geluActivation) {
            trace("reject:unsupported_activation:" + activationType, config, input, gateW, upW);
            return null;
        }
        if (gateB != null || upB != null) {
            trace("reject:bias_present", config, input, gateW, upW);
            return null;
        }
        if (combinedBuffer == null || combinedBuffer.isClosed()) {
            trace("reject:combined_workspace_unavailable", config, input, gateW, upW);
            return null;
        }
        if (!metalLinearEnabled || metalBinding == null) {
            trace("reject:metal_unavailable", config, input, gateW, upW);
            return null;
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, gateW, traits, PROFILE_KEY)
                || !DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, upW, traits, PROFILE_KEY)) {
            trace("reject:candidate_ineligible", config, input, gateW, upW);
            return null;
        }
        if (gateW.rank() != 2
                || upW.rank() != 2
                || gateW.size(0) != upW.size(0)
                || gateW.size(1) != upW.size(1)) {
            trace("reject:shape_mismatch", config, input, gateW, upW);
            return null;
        }
        long inputDim = input.size(-1);
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows != 1L) {
            trace("reject:not_single_token_rows:" + rows, config, input, gateW, upW);
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
                upW, traits, PROFILE_KEY, allowBf16ToF16Weights);
        AccelTensor metalGateW = toMetalHalfWeight(gateW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        AccelTensor metalUpW = toMetalHalfWeight(upW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        if (metalGateW == null || metalUpW == null
                || metalGateW.quantType() != metalUpW.quantType()
                || (metalGateW.quantType() != AccelTensor.QuantType.F16
                        && metalGateW.quantType() != AccelTensor.QuantType.BF16)) {
            trace("reject:weight_conversion_failed:native_bf16=" + nativeBf16Weights, config, input, gateW, upW);
            return null;
        }
        if (nativeBf16Weights && siluActivation && !capabilities.supportsSwigluGateUpMatvecBf16()) {
            trace("reject:swiglu_bf16_symbol_unavailable", config, input, gateW, upW);
            return null;
        }
        if (nativeBf16Weights && geluActivation && !capabilities.supportsGegluGateUpMatvecBf16()) {
            trace("reject:geglu_bf16_symbol_unavailable", config, input, gateW, upW);
            return null;
        }
        if (!nativeBf16Weights && siluActivation && !capabilities.supportsSwigluGateUpMatvecHalf()) {
            trace("reject:swiglu_symbol_unavailable", config, input, gateW, upW);
            return null;
        }
        if (!nativeBf16Weights && geluActivation && !capabilities.supportsGegluGateUpMatvecHalf()) {
            trace("reject:geglu_symbol_unavailable", config, input, gateW, upW);
            return null;
        }

        long intermediateDim = gateW.size(0);
        if (!combinedBuffer.hasShape(input.shapeWithLastDim(intermediateDim))) {
            trace("reject:combined_shape_mismatch", config, input, gateW, upW);
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        try {
            int rc;
            if (nativeBf16Weights && siluActivation) {
                rc = metalBinding.swigluGateUpMatvecBf16(
                        combinedBuffer.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim));
            } else if (nativeBf16Weights) {
                rc = metalBinding.gegluGateUpMatvecBf16(
                        combinedBuffer.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim));
            } else if (siluActivation) {
                rc = metalBinding.swigluGateUpMatvecHalf(
                        combinedBuffer.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim));
            } else {
                rc = metalBinding.gegluGateUpMatvecHalf(
                        combinedBuffer.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim));
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal gated gate/up matvec failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY,
                    (siluActivation ? "swiglu" : "geglu") + "_gate_up_matvec"
                            + (nativeBf16Weights ? "_bf16" : "_f16"));
            DirectInferenceProfiler.recordLinearNanos(PROFILE_KEY, System.nanoTime() - t0);
            trace("accept:" + (siluActivation ? "swiglu" : "geglu") + ":native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW);
            return combinedBuffer;
        } catch (RuntimeException e) {
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW);
            log.debugf("Falling back from Metal gated gate/up matvec: %s", e.getMessage());
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
                              AccelTensor upW) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, gateW, upW, null);
    }
}
