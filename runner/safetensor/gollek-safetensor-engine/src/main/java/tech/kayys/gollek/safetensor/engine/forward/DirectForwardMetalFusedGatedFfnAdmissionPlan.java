/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.FFNActivationType;

record DirectForwardMetalFusedGatedFfnAdmissionPlan(
        DirectForwardMetalFfnActivationPlan activation,
        long rows,
        String rejectionReason) {

    static DirectForwardMetalFusedGatedFfnAdmissionPlan from(
            MetalBinding metalBinding,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        return from(
                metalBinding != null,
                capabilities,
                traits,
                metalLinearEnabled,
                input,
                activationType,
                gateB,
                upB,
                downB);
    }

    static DirectForwardMetalFusedGatedFfnAdmissionPlan from(
            boolean metalBindingAvailable,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        DirectForwardMetalFfnActivationPlan activation =
                DirectForwardMetalFfnActivationPlan.from(activationType);
        if (DirectForwardFfnFastPathPolicy.isMetalFusedFfnDisabled()) {
            return reject(activation, "disabled");
        }
        if (!activation.supported()) {
            return reject(activation, activation.unsupportedReason());
        }
        if (gateB != null || upB != null || downB != null) {
            return reject(activation, "bias_present");
        }
        if (!metalLinearEnabled) {
            return reject(activation, "metal_linear_disabled");
        }
        if (!metalBindingAvailable) {
            return reject(activation, "metal_binding_unavailable");
        }
        if (input == null) {
            return reject(activation, "input_null");
        }
        boolean gemma4PolicyTarget = DirectForwardElementwisePolicy.isGemma4FfnPolicyTarget(traits);
        if (activation.gelu() && !DirectForwardFfnFastPathPolicy.shouldUseMetalGegluFusedFfn(traits)) {
            return reject(activation, "geglu_flag_disabled");
        }
        String capabilityRejection = activation.fusedHalfCapabilityRejection(capabilities);
        if (capabilityRejection != null) {
            return reject(activation, capabilityRejection);
        }
        long rows = DirectForwardMetalFfnShapePlan.rows(input);
        if (rows <= 0L) {
            return reject(activation, rows, "invalid_rows:" + rows);
        }
        if (activation.silu() && traits.qwenText() && !DirectForwardFfnFastPathPolicy.shouldUseQwenMetalFusedFfn()) {
            return reject(
                    activation,
                    rows,
                    rows == 1L ? "qwen_decode_pair_path_preferred" : "qwen_prefill_pair_path_preferred");
        }
        if (gemma4PolicyTarget && !DirectForwardFfnFastPathPolicy.allowGemma4FusedHalfFfn(rows, traits)) {
            return reject(
                    activation,
                    rows,
                    rows == 1L ? "gemma4_decode_flag_disabled" : "gemma4_prefill_flag_disabled");
        }
        if (rows != 1L && !DirectForwardFfnFastPathPolicy.shouldUseMetalFusedFfnPrefill(traits)) {
            return reject(activation, rows, "prefill_flag_disabled:rows=" + rows);
        }
        return new DirectForwardMetalFusedGatedFfnAdmissionPlan(activation, rows, null);
    }

    boolean admitted() {
        return rejectionReason == null;
    }

    String rejectionDecision() {
        return admitted() ? null : "reject:" + rejectionReason;
    }

    private static DirectForwardMetalFusedGatedFfnAdmissionPlan reject(
            DirectForwardMetalFfnActivationPlan activation,
            String reason) {
        return reject(activation, -1L, reason);
    }

    private static DirectForwardMetalFusedGatedFfnAdmissionPlan reject(
            DirectForwardMetalFfnActivationPlan activation,
            long rows,
            String reason) {
        return new DirectForwardMetalFusedGatedFfnAdmissionPlan(activation, rows, reason);
    }
}
