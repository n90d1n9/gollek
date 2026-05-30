/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

final class FlashAttentionProjector {
    private final MetalBinding metalBinding;
    private final BooleanSupplier canUseMetal;

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal) {
        this.metalBinding = metalBinding;
        this.canUseMetal = canUseMetal;
    }

    AccelTensor attentionOutputBufferView(AttentionInput in, AccelTensor projectedInput) {
        if (in.attentionOutputBuffer == null || projectedInput == null || in.oW == null || in.oW.rank() != 2) {
            return null;
        }
        return AccelTensor.view(in.attentionOutputBuffer, projectedInput.shapeWithLastDim(in.oW.size(0)));
    }

    ProjectionBuffers attentionProjectionBuffers(AttentionInput in, boolean enabled) {
        if (!enabled || in == null || in.kvCache == null || in.qW == null || in.kW == null || in.vW == null) {
            return null;
        }
        MemorySegment scratch = in.kvCache.getWorkspace().getCombinedSeg();
        if (scratch == null || in.x == null || in.x.rank() < 2) {
            return null;
        }
        long qDim = in.qW.size(0);
        long kDim = in.kW.size(0);
        long vDim = in.vW.size(0);
        if (qDim <= 0 || kDim <= 0 || vDim <= 0) {
            return null;
        }
        long batchTokens = in.x.numel() / Math.max(1L, in.x.size(-1));
        if (batchTokens <= 0L) {
            return null;
        }
        long qElements = Math.multiplyExact(batchTokens, qDim);
        long kElements = Math.multiplyExact(batchTokens, kDim);
        long vElements = Math.multiplyExact(batchTokens, vDim);
        long qBytes = Math.multiplyExact(qElements, (long) Float.BYTES);
        long kBytes = Math.multiplyExact(kElements, (long) Float.BYTES);
        long vBytes = Math.multiplyExact(vElements, (long) Float.BYTES);
        long requiredBytes = Math.addExact(Math.addExact(qBytes, kBytes), vBytes);
        if (scratch.byteSize() < requiredBytes) {
            return null;
        }
        return new ProjectionBuffers(
                AccelTensor.view(scratch.asSlice(0, qBytes), in.x.shapeWithLastDim(qDim)),
                AccelTensor.view(scratch.asSlice(qBytes, kBytes), in.x.shapeWithLastDim(kDim)),
                AccelTensor.view(scratch.asSlice(qBytes + kBytes, vBytes), in.x.shapeWithLastDim(vDim)));
    }

    AccelTensor project(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy) {
        return project(input, weight, bias, profileKey, config, modelPolicy, null);
    }

    LinearTriple projectPackedQkv(AttentionInput in, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            FlashAttentionHeadLayout layout) {
        if (in == null || in.arch == null || !in.arch.hasFusedQKV() || in.qW == null || layout == null) {
            return null;
        }
        if (in.qW.size(0) != layout.packedQkvProjectionDim()) {
            throw new IllegalArgumentException(
                    "Packed QKV projection rows do not match attention layout: rows=" + in.qW.size(0)
                            + " expected=" + layout.packedQkvProjectionDim()
                            + " qHeads=" + layout.numQueryHeads()
                            + " kvHeads=" + layout.numKeyValueHeads()
                            + " headDim=" + layout.headDim());
        }

        AccelTensor packed = project(in.x, in.qW, in.qB, "attn_qkv_proj_packed", config, modelPolicy);
        try {
            if (!layout.matchesPackedProjection(packed)) {
                throw new IllegalArgumentException(
                        "Packed QKV projection output does not match attention layout: lastDim=" + packed.size(-1)
                                + " expected=" + layout.packedQkvProjectionDim());
            }
            long qEnd = layout.queryProjectionDim();
            long kEnd = qEnd + layout.keyValueProjectionDim();
            long vEnd = kEnd + layout.keyValueProjectionDim();
            return new LinearTriple(
                    materializedLastDimSlice(packed, 0, qEnd),
                    materializedLastDimSlice(packed, qEnd, kEnd),
                    materializedLastDimSlice(packed, kEnd, vEnd));
        } finally {
            if (packed != null && !packed.isClosed()) {
                packed.close();
            }
        }
    }

    AccelTensor project(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, AccelTensor outputBuffer) {
        long t0 = System.nanoTime();
        AccelTensor projected = tryMetalHalfLinear(input, weight, bias, config, modelPolicy, profileKey,
                outputBuffer);
        if (projected == null) {
            projected = tryMetalFloatLinear(input, weight, bias);
            if (projected != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "metal_float_matmul");
            }
        }
        if (projected == null) {
            projected = AccelOps.linear(input, weight, bias);
            DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear");
        }
        DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
        return projected;
    }

    private AccelTensor materializedLastDimSlice(AccelTensor packed, long start, long end) {
        AccelTensor view = packed.slice(-1, start, end);
        AccelTensor contiguous = view.contiguous();
        if (contiguous == view) {
            AccelTensor copy = AccelTensor.copyOf(view.dataPtr(), view.shape());
            view.close();
            return copy;
        }
        view.close();
        return contiguous;
    }

    LinearPair tryMetalHalfLinearPairMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy) {
        if (!FlashAttentionRuntimeOptions.metalMixedHalfLinearPairEnabled()
                || !canUseExperimentalMetalLinear()
                || !metalBinding.supportsMatmulTransposedRightHalfPairMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !canUseMixedHalfPairWeight(firstWeight, modelPolicy)
                || !canUseMixedHalfPairWeight(secondWeight, modelPolicy)) {
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(modelPolicy, firstWeight, secondWeight);
        AccelTensor firstMetalWeight = toMetalHalfWeight(firstWeight, nativeBf16Weights, modelPolicy);
        AccelTensor secondMetalWeight = toMetalHalfWeight(secondWeight, nativeBf16Weights, modelPolicy);
        if (firstMetalWeight == null || secondMetalWeight == null) {
            return null;
        }

        long[] inputShape = input.shape();
        long[] firstWeightShape = firstMetalWeight.shape();
        long[] secondWeightShape = secondMetalWeight.shape();
        if (inputShape.length < 2
                || firstWeightShape.length != 2
                || secondWeightShape.length != 2
                || firstWeightShape[1] != secondWeightShape[1]) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L || k != firstWeightShape[1]) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long[] firstOutputShape = inputShape.clone();
        firstOutputShape[firstOutputShape.length - 1] = firstWeightShape[0];
        long[] secondOutputShape = inputShape.clone();
        secondOutputShape[secondOutputShape.length - 1] = secondWeightShape[0];
        AccelTensor first = AccelTensor.zeros(firstOutputShape);
        AccelTensor second = AccelTensor.zeros(secondOutputShape);

        try {
            int rc = metalBinding.matmulTransposedRightHalfPairMixed(
                    first.dataPtr(),
                    second.dataPtr(),
                    contiguousInput.dataPtr(),
                    firstMetalWeight.dataPtr(),
                    secondMetalWeight.dataPtr(),
                    Math.toIntExact(rows),
                    Math.toIntExact(k),
                    Math.toIntExact(firstWeightShape[0]),
                    Math.toIntExact(secondWeightShape[0]),
                    1.0f,
                    0.0f,
                    nativeBf16Weights);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPairMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceProfiler.recordLinearPath(profileKey,
                    nativeBf16Weights ? "mixed_pair_matmul_bf16" : "mixed_pair_matmul_f16");
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new LinearPair(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    LinearTriple tryMetalHalfLinearTripleMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            AccelTensor thirdWeight,
            AccelTensor thirdBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer,
            AccelTensor thirdOutputBuffer) {
        if (FlashAttentionRuntimeOptions.disableMetalMixedHalfLinearTriple()
                || !FlashAttentionRuntimeOptions.metalMixedHalfLinearPairEnabled()
                || !canUseExperimentalMetalLinear()
                || !metalBinding.supportsMatmulTransposedRightHalfTripleMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !canUseMixedHalfPairWeight(firstWeight, modelPolicy)
                || !canUseMixedHalfPairWeight(secondWeight, modelPolicy)
                || !canUseMixedHalfPairWeight(thirdWeight, modelPolicy)) {
            return null;
        }

        long[] inputShape = input.shape();
        if (inputShape.length < 2) {
            return null;
        }
        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(modelPolicy, firstWeight, secondWeight,
                thirdWeight);
        AccelTensor firstMetalWeight = toMetalHalfWeight(firstWeight, nativeBf16Weights, modelPolicy);
        AccelTensor secondMetalWeight = toMetalHalfWeight(secondWeight, nativeBf16Weights, modelPolicy);
        AccelTensor thirdMetalWeight = toMetalHalfWeight(thirdWeight, nativeBf16Weights, modelPolicy);
        if (firstMetalWeight == null || secondMetalWeight == null || thirdMetalWeight == null) {
            return null;
        }

        long[] firstWeightShape = firstMetalWeight.shape();
        long[] secondWeightShape = secondMetalWeight.shape();
        long[] thirdWeightShape = thirdMetalWeight.shape();
        if (firstWeightShape.length != 2
                || secondWeightShape.length != 2
                || thirdWeightShape.length != 2
                || firstWeightShape[1] != k
                || secondWeightShape[1] != k
                || thirdWeightShape[1] != k) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long[] firstOutputShape = inputShape.clone();
        firstOutputShape[firstOutputShape.length - 1] = firstWeightShape[0];
        long[] secondOutputShape = inputShape.clone();
        secondOutputShape[secondOutputShape.length - 1] = secondWeightShape[0];
        long[] thirdOutputShape = inputShape.clone();
        thirdOutputShape[thirdOutputShape.length - 1] = thirdWeightShape[0];
        AccelTensor first = reusableOutputTensor(firstOutputBuffer, firstOutputShape);
        AccelTensor second = reusableOutputTensor(secondOutputBuffer, secondOutputShape);
        AccelTensor third = reusableOutputTensor(thirdOutputBuffer, thirdOutputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n0 = Math.toIntExact(firstWeightShape[0]);
            int n1 = Math.toIntExact(secondWeightShape[0]);
            int n2 = Math.toIntExact(thirdWeightShape[0]);
            int rc = -2;
            String executionPath = nativeBf16Weights ? "mixed_triple_matmul_bf16" : "mixed_triple_matmul_f16";
            if (m == 1
                    && nativeBf16Weights
                    && FlashAttentionRuntimeOptions.shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightBf16TripleMixed()) {
                rc = metalBinding.matvecTransposedRightBf16TripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
                if (rc == 0) {
                    executionPath = "mixed_triple_matvec_bf16";
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weights
                    && FlashAttentionRuntimeOptions.shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightHalfTripleMixed()) {
                rc = metalBinding.matvecTransposedRightHalfTripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
                if (rc == 0) {
                    executionPath = "mixed_triple_matvec_f16";
                }
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfTripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        m,
                        kk,
                        n0,
                        n1,
                        n2,
                        1.0f,
                        0.0f,
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfTripleMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            AccelTensor thirdOut = addBiasIfNeeded(third, thirdBias);
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new LinearTriple(firstOut, secondOut, thirdOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            third.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private boolean canUseMixedHalfPairWeight(AccelTensor weight, FlashAttentionModelPolicy modelPolicy) {
        if (weight == null || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && modelPolicy.disallowBf16ToF16LinearConversion()) {
            return shouldUseNativeMetalBf16Linear(modelPolicy, weight);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(modelPolicy));
    }

    private AccelTensor toMetalHalfWeight(AccelTensor weight, boolean nativeBf16,
            FlashAttentionModelPolicy modelPolicy) {
        if (weight == null) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.F16) {
            return weight;
        }
        if (nativeBf16 && weight.quantType() == AccelTensor.QuantType.BF16) {
            return weight;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16
                && modelPolicy.disallowBf16ToF16LinearConversion()) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(modelPolicy)) {
            return weight.toF16CachedUpTo(FlashAttentionRuntimeOptions.metalF16WeightCacheMaxBytes());
        }
        return null;
    }

    private AccelTensor addBiasIfNeeded(AccelTensor tensor, AccelTensor bias) {
        if (bias == null) {
            return tensor;
        }
        AccelTensor biased = AccelOps.add(tensor, bias);
        tensor.close();
        return biased;
    }

    private boolean canUseExperimentalMetalLinear() {
        return metalBinding != null
                && canUseMetal.getAsBoolean()
                && FlashAttentionRuntimeOptions.experimentalMetalLinearEnabled();
    }

    private boolean allowMetalBf16Linear(FlashAttentionModelPolicy modelPolicy) {
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            if (FlashAttentionRuntimeOptions.disableGemma4MetalBf16Linear()) {
                return false;
            }
        }
        return FlashAttentionRuntimeOptions.allowMetalBf16Linear();
    }

    private boolean preferNativeMetalBf16Linear(FlashAttentionModelPolicy modelPolicy) {
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            String explicit = FlashAttentionRuntimeOptions.enableGemma4MetalBf16LinearValue();
            if (explicit != null && !explicit.isBlank()) {
                return Boolean.parseBoolean(explicit) && allowMetalBf16Linear(modelPolicy);
            }
            return allowMetalBf16Linear(modelPolicy);
        }
        return false;
    }

    private boolean shouldUseNativeMetalBf16Linear(FlashAttentionModelPolicy modelPolicy, AccelTensor... weights) {
        if (!preferNativeMetalBf16Linear(modelPolicy) || weights == null || weights.length == 0) {
            return false;
        }
        for (AccelTensor weight : weights) {
            if (weight == null || weight.quantType() != AccelTensor.QuantType.BF16) {
                return false;
            }
        }
        return true;
    }

    private AccelTensor tryMetalHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, String profileKey, AccelTensor outputBuffer) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        boolean nativeBf16Weight = shouldUseNativeMetalBf16Linear(modelPolicy, weight);
        if (quantType == AccelTensor.QuantType.BF16
                && modelPolicy.disallowBf16ToF16LinearConversion()
                && !nativeBf16Weight) {
            return null;
        }
        if (quantType != AccelTensor.QuantType.F16
                && (quantType != AccelTensor.QuantType.BF16 || !allowMetalBf16Linear(modelPolicy))) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }
        long[] inputShape = input.shape();
        long[] weightShape = weight.shape();
        if (inputShape.length < 2 || weightShape.length != 2) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = weightShape[0];
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(weightShape[0]);
            int rc = -2;
            String executionPath = nativeBf16Weight ? "metal_matmul_bf16" : "metal_matmul_f16";
            if (m == 1
                    && nativeBf16Weight
                    && FlashAttentionRuntimeOptions.shouldUseMetalHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedRightBf16()) {
                rc = metalBinding.matvecTransposedRightBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        weight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "bf16_matvec";
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && FlashAttentionRuntimeOptions.shouldUseMetalTransposedHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedWeightHalf()) {
                AccelTensor transposedWeight = weight.toF16Transposed2dCachedUpTo(
                        FlashAttentionRuntimeOptions.metalF16WeightCacheMaxBytes());
                if (transposedWeight != null
                        && transposedWeight.size(0) == k
                        && transposedWeight.size(1) == weightShape[0]) {
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
            AccelTensor metalWeight = null;
            if (rc != 0) {
                metalWeight = toMetalHalfWeight(weight, nativeBf16Weight, modelPolicy);
                if (metalWeight == null) {
                    return null;
                }
            }
            if (m == 1
                    && !nativeBf16Weight
                    && FlashAttentionRuntimeOptions.shouldUseMetalHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedRightHalf()) {
                rc = metalBinding.matvecTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "matvec";
                }
            }
            if (rc != 0 && metalWeight != null) {
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
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor reusableOutputTensor(AccelTensor outputBuffer, long[] outputShape) {
        if (outputBuffer != null
                && !outputBuffer.isClosed()
                && outputBuffer.hasShape(outputShape)) {
            return outputBuffer;
        }
        return AccelTensor.zeros(outputShape);
    }

    private AccelTensor tryMetalFloatLinear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        if (input == null || weight == null) {
            return null;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }

        long[] inputShape = input.shape();
        long[] weightShape = weight.shape();
        if (inputShape.length < 2 || weightShape.length != 2) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = weightShape[0];
        AccelTensor out = AccelTensor.zeros(outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(weightShape[0]);
            int rc = metalBinding.matmulTransposedRight(
                    out.dataPtr(),
                    contiguousInput.dataPtr(),
                    weight.dataPtr(),
                    m, kk, n,
                    1.0f, 0.0f);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRight failed with code " + rc);
            }
            if (bias == null) {
                return out;
            }
            AccelTensor biased = AccelOps.add(out, bias);
            out.close();
            return biased;
        } catch (RuntimeException e) {
            out.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    record LinearPair(AccelTensor first, AccelTensor second) {
    }

    record LinearTriple(AccelTensor first, AccelTensor second, AccelTensor third) {
    }

    record ProjectionBuffers(AccelTensor q, AccelTensor k, AccelTensor v) {
    }
}
