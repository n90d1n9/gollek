/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.addBiasIfNeeded;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalHalfLinear {
    private DirectForwardMetalHalfLinear() {
    }

    static AccelTensor tryLinear(Logger log,
                                 MetalBinding metalBinding,
                                 DirectForwardMetalCapabilities capabilities,
                                 ModelConfigTraits traits,
                                 ModelConfig config,
                                 boolean metalLinearEnabled,
                                 boolean decodeLogitsPhase,
                                 AccelTensor input,
                                 AccelTensor weight,
                                 AccelTensor bias,
                                 AccelTensor outputBuffer,
                                 String profileKey) {
        DirectForwardMetalHalfLinearAdmissionPlan admissionPlan =
                DirectForwardMetalHalfLinearAdmissionPlan.from(
                        traits,
                        metalLinearEnabled,
                        input,
                        weight,
                        profileKey);
        if (!admissionPlan.admitted()) {
            return null;
        }
        DirectForwardMetalLinearShapePlan shapePlan =
                DirectForwardMetalLinearShapePlan.single(input, weight);
        if (shapePlan == null) {
            return null;
        }
        DirectForwardMetalLinearWeightPlan weightPlan = DirectForwardMetalLinearWeightPlan.single(
                traits,
                profileKey,
                decodeLogitsPhase,
                shapePlan.rows(),
                weight);
        AccelTensor out = reusableOutputTensor(outputBuffer, shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int m = Math.toIntExact(shapePlan.rows());
            int kk = Math.toIntExact(shapePlan.inputDim());
            int n = Math.toIntExact(shapePlan.outputDim());
            DirectForwardMetalHalfLinearExecutionPlan executionPlan =
                    DirectForwardMetalHalfLinearExecutionPlan.from(
                            m,
                            kk,
                            n,
                            weightPlan.nativeBf16Weights(),
                            capabilities,
                            DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvec(
                                    traits, config, n, profileKey),
                            DirectForwardMetalHalfMatvecPolicy.shouldUseMetalLogitsMpsMatvec(
                                    traits, n, kk, profileKey),
                            DirectForwardMetalHalfMatvecPolicy.shouldUseMetalTransposedHalfMatvec(
                                    traits, n, profileKey));
            int rc = -2;
            String executionPath = executionPlan.matmulPath();
            AccelTensor metalWeight = null;
            if (executionPlan.nativeBf16MatvecCandidate()) {
                metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                        weight, traits, true, weightPlan.allowBf16ToF16Weights());
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightBf16(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.nativeBf16MatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.mpsHalfMatvecCandidate()) {
                metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                        weight, traits, false, weightPlan.allowBf16ToF16Weights());
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalfMps(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.mpsHalfMatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.transposedHalfMatvecCandidate()) {
                AccelTensor transposedWeight = DirectForwardLinearCachePolicy.cachedTransposedF16Weight(weight);
                if (executionPlan.matchesTransposedWeight(transposedWeight)) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.transposedHalfMatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.halfMatvecCandidate()) {
                if (metalWeight == null) {
                    metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                            weight, traits, false, weightPlan.allowBf16ToF16Weights());
                }
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalf(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.halfMatvecPath();
                    }
                }
            }
            if (rc != 0) {
                if (metalWeight == null) {
                    metalWeight = weightPlan.weight();
                }
                if (metalWeight == null) {
                    out.close();
                    return null;
                }
                rc = metalBinding.matmulTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.tensor().dataPtr(),
                        metalWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        executionPlan.nativeBf16Weight());
            }
            if (rc != 0) {
                log.infof("Metal linear failed with rc=%d, m=%d, halfMatvecCandidate=%b, nativeBf16Weight=%b", rc, m, executionPlan.halfMatvecCandidate(), executionPlan.nativeBf16Weight());
                throw new IllegalStateException("Metal matmulTransposedRightHalf failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            return addBiasIfNeeded(out, bias);
        } catch (RuntimeException e) {
            out.close();
            System.err.println("Falling back from Metal half linear to AccelOps: " + e.getMessage());
            return null;
        }
    }
}
