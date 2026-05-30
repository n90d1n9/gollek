/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardNativeBf16MatvecPolicy.describeNativeBf16MatvecPath;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
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
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled,
                input,
                weight,
                traits,
                profileKey)) {
            return null;
        }
        long k = input.size(-1);
        long rows = input.numel() / Math.max(1L, k);
        boolean allowBf16ToF16Weight = DirectForwardMetalLinearPolicy.allowGemma4Bf16ToF16LinearForRows(
                rows,
                traits,
                profileKey,
                decodeLogitsPhase);
        boolean nativeBf16Weight = DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                weight,
                traits,
                profileKey,
                allowBf16ToF16Weight);
        AccelTensor contiguousInput = input.contiguous();
        long outputDim = weight.size(0);
        long[] outputShape = input.shapeWithLastDim(outputDim);
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(outputDim);
            int rc = -2;
            String executionPath = "metal_matmul";
            AccelTensor metalWeight = null;
            if (m == 1
                    && nativeBf16Weight
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvec(
                            traits, config, n, profileKey)
                    && capabilities.supportsMatvecTransposedRightBf16()) {
                metalWeight = toMetalHalfWeight(weight, traits, true, profileKey, allowBf16ToF16Weight);
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightBf16(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = describeNativeBf16MatvecPath(kk, n);
                    }
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalLogitsMpsMatvec(
                            traits, n, kk, profileKey)
                    && capabilities.supportsMatvecTransposedRightHalfMps()) {
                metalWeight = toMetalHalfWeight(weight, traits, false, profileKey, allowBf16ToF16Weight);
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalfMps(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = "mps_matvec";
                    }
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalTransposedHalfMatvec(
                            traits, n, profileKey)
                    && capabilities.supportsMatvecTransposedWeightHalf()) {
                AccelTensor transposedWeight = DirectForwardLinearCachePolicy.cachedTransposedF16Weight(weight);
                if (transposedWeight != null
                        && transposedWeight.size(0) == k
                        && transposedWeight.size(1) == outputDim) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = "transposed_matvec";
                    }
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvec(
                            traits, config, n, profileKey)
                    && capabilities.supportsMatvecTransposedRightHalf()) {
                if (metalWeight == null) {
                    metalWeight = toMetalHalfWeight(weight, traits, false, profileKey, allowBf16ToF16Weight);
                }
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalf(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = "matvec";
                    }
                }
            }
            if (rc != 0) {
                if (metalWeight == null) {
                    metalWeight = toMetalHalfWeight(
                            weight,
                            traits,
                            nativeBf16Weight,
                            profileKey,
                            allowBf16ToF16Weight);
                }
                if (metalWeight == null) {
                    out.close();
                    return null;
                }
                rc = metalBinding.matmulTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        nativeBf16Weight);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalf failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            if (bias == null) {
                return out;
            }
            AccelTensor biased = AccelOps.add(out, bias);
            out.close();
            return biased;
        } catch (RuntimeException e) {
            out.close();
            log.debugf("Falling back from Metal half linear to AccelOps: %s", e.getMessage());
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
                                                 String profileKey,
                                                 boolean allowBf16ToF16) {
        return DirectForwardLinearCachePolicy.toMetalHalfWeight(
                weight,
                nativeBf16,
                traits.gemma4Text(),
                allowBf16ToF16,
                DirectForwardMetalLinearPolicy.allowMetalBf16Linear(traits));
    }
}
