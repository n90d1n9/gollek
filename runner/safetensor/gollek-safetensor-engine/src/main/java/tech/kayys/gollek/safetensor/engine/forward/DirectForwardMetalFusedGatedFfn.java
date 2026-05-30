/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalFusedGatedFfn {
    private static final String PATH = "fused-gated-ffn";
    private static final String PROFILE_KEY = "ffn_fused_metal";

    private DirectForwardMetalFusedGatedFfn() {
    }

    static String skipReason(MetalBinding metalBinding,
                             DirectForwardMetalCapabilities capabilities,
                             ModelConfigTraits traits,
                             boolean metalLinearEnabled,
                             AccelTensor input,
                             FFNActivationType activationType,
                             AccelTensor gateB,
                             AccelTensor upB,
                             AccelTensor downB) {
        boolean siluActivation = activationType == FFNActivationType.SILU;
        boolean geluActivation = activationType == FFNActivationType.GELU;
        if (DirectForwardFfnFastPathPolicy.isMetalFusedFfnDisabled()) {
            return "disabled";
        }
        if (!siluActivation && !geluActivation) {
            return "unsupported_activation:" + activationType;
        }
        if (gateB != null || upB != null || downB != null) {
            return "bias_present";
        }
        if (!metalLinearEnabled) {
            return "metal_linear_disabled";
        }
        if (metalBinding == null) {
            return "metal_binding_unavailable";
        }
        if (input == null) {
            return "input_null";
        }
        boolean gemma4PolicyTarget = DirectForwardElementwisePolicy.isGemma4FfnPolicyTarget(traits);
        if (geluActivation && !DirectForwardFfnFastPathPolicy.shouldUseMetalGegluFusedFfn(traits)) {
            return "geglu_flag_disabled";
        }
        if (siluActivation && !capabilities.supportsSwigluFfnHalf()) {
            return "swiglu_symbol_unavailable";
        }
        if (geluActivation && !capabilities.supportsGegluFfnHalf()) {
            return "geglu_symbol_unavailable";
        }
        long inputDim = input.size(-1);
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows <= 0L) {
            return "invalid_rows:" + rows;
        }
        if (siluActivation && traits.qwenText() && !DirectForwardFfnFastPathPolicy.shouldUseQwenMetalFusedFfn()) {
            return rows == 1L ? "qwen_decode_pair_path_preferred" : "qwen_prefill_pair_path_preferred";
        }
        if (gemma4PolicyTarget && !DirectForwardFfnFastPathPolicy.allowGemma4FusedHalfFfn(rows, traits)) {
            return rows == 1L ? "gemma4_decode_flag_disabled" : "gemma4_prefill_flag_disabled";
        }
        if (rows != 1L && !DirectForwardFfnFastPathPolicy.shouldUseMetalFusedFfnPrefill(traits)) {
            return "prefill_flag_disabled:rows=" + rows;
        }
        return null;
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
        if (DirectForwardFfnFastPathPolicy.isMetalFusedFfnDisabled()) {
            return reject("disabled", config, input, gateW, upW, downW);
        }
        if (!siluActivation && !geluActivation) {
            return reject("unsupported_activation:" + activationType, config, input, gateW, upW, downW);
        }
        boolean gemma4PolicyTarget = DirectForwardElementwisePolicy.isGemma4FfnPolicyTarget(traits);
        if (geluActivation && !DirectForwardFfnFastPathPolicy.shouldUseMetalGegluFusedFfn(traits)) {
            return reject("geglu_flag_disabled", config, input, gateW, upW, downW);
        }
        if (gateB != null || upB != null || downB != null) {
            return reject("bias_present", config, input, gateW, upW, downW);
        }
        if (!metalLinearEnabled) {
            return reject("metal_linear_disabled", config, input, gateW, upW, downW);
        }
        if (metalBinding == null) {
            return reject("metal_binding_unavailable", config, input, gateW, upW, downW);
        }
        if (siluActivation && !capabilities.supportsSwigluFfnHalf()) {
            return reject("swiglu_symbol_unavailable", config, input, gateW, upW, downW);
        }
        if (geluActivation && !capabilities.supportsGegluFfnHalf()) {
            return reject("geglu_symbol_unavailable", config, input, gateW, upW, downW);
        }
        if (input == null) {
            return reject("input_null", config, input, gateW, upW, downW);
        }
        long inputDim = input.size(-1);
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows <= 0L) {
            return reject("invalid_rows:" + rows, config, input, gateW, upW, downW);
        }
        if (siluActivation && traits.qwenText() && !DirectForwardFfnFastPathPolicy.shouldUseQwenMetalFusedFfn()) {
            return reject(
                    rows == 1L ? "qwen_decode_pair_path_preferred" : "qwen_prefill_pair_path_preferred",
                    config, input, gateW, upW, downW);
        }
        if (gemma4PolicyTarget && !DirectForwardFfnFastPathPolicy.allowGemma4FusedHalfFfn(rows, traits)) {
            return reject(rows == 1L ? "gemma4_decode_flag_disabled" : "gemma4_prefill_flag_disabled",
                    config, input, gateW, upW, downW);
        }
        if (rows != 1L && !DirectForwardFfnFastPathPolicy.shouldUseMetalFusedFfnPrefill(traits)) {
            return reject("prefill_flag_disabled:rows=" + rows, config, input, gateW, upW, downW);
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, gateW, traits, PROFILE_KEY)) {
            return reject("gate_candidate_ineligible", config, input, gateW, upW, downW);
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, upW, traits, PROFILE_KEY)) {
            return reject("up_candidate_ineligible", config, input, gateW, upW, downW);
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfWeight(downW, traits, PROFILE_KEY)) {
            return reject("down_weight_ineligible", config, input, gateW, upW, downW);
        }

        if (gateW.rank() != 2
                || upW.rank() != 2
                || downW.rank() != 2
                || gateW.size(0) != upW.size(0)
                || gateW.size(1) != upW.size(1)
                || downW.size(1) != gateW.size(0)) {
            return reject("shape_mismatch", config, input, gateW, upW, downW);
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
        AccelTensor metalGateW = toMetalHalfWeight(gateW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        AccelTensor metalUpW = toMetalHalfWeight(upW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        AccelTensor metalDownW = toMetalHalfWeight(downW, traits, nativeBf16Weights, allowBf16ToF16Weights);
        if (metalGateW == null || metalUpW == null || metalDownW == null) {
            return reject("weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long intermediateDim = metalGateW.size(0);
        long outputDim = metalDownW.size(0);
        long[] outputShape = input.shapeWithLastDim(outputDim);
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int rc;
            if (siluActivation) {
                rc = metalBinding.swigluFfnHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(rows),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim),
                        nativeBf16Weights);
            } else {
                rc = metalBinding.gegluFfnHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(rows),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim),
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal fused gated FFN failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY,
                    (siluActivation ? "swiglu" : "geglu") + "_fused"
                            + (nativeBf16Weights ? "_bf16" : "_f16"));
            DirectInferenceProfiler.recordLinearNanos(
                    siluActivation ? "ffn_fused_metal" : "ffn_geglu_fused_metal",
                    System.nanoTime() - t0);
            trace("accept:" + (siluActivation ? "swiglu" : "geglu") + ":native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW, downW);
            log.debugf("Falling back from fused Metal gated FFN: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static AccelTensor reject(String reason,
                                      ModelConfig config,
                                      AccelTensor input,
                                      AccelTensor gateW,
                                      AccelTensor upW,
                                      AccelTensor downW) {
        trace("reject:" + reason, config, input, gateW, upW, downW);
        return null;
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
