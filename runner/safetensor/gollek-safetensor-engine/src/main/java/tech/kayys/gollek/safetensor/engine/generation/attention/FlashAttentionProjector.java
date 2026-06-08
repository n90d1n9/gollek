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
    private final FlashAttentionMetalLinear metalLinear;

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal) {
        this(metalBinding, canUseMetal, FlashAttentionLinearOptions.defaults());
    }

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions) {
        this(metalBinding, canUseMetal, linearOptions, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions, FlashAttentionMatvecOptions matvecOptions) {
        this(new FlashAttentionMetalLinear(metalBinding, canUseMetal, linearOptions, matvecOptions));
    }

    FlashAttentionProjector(FlashAttentionMetalLinear metalLinear) {
        this.metalLinear = metalLinear;
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
        if (in == null || in.qW == null || layout == null || !layout.packedQkvProjection()) {
            return null;
        }
        FlashAttentionShapeAdmissionPlan weightAdmission =
                FlashAttentionShapeAdmissionPlan.packedQkvWeight(in.qW, layout);
        if (!weightAdmission.admitted()) {
            throw weightAdmission.asException();
        }

        AccelTensor packed = project(in.x, in.qW, in.qB, "attn_qkv_proj_packed", config, modelPolicy);
        try {
            FlashAttentionShapeAdmissionPlan outputAdmission =
                    FlashAttentionShapeAdmissionPlan.packedQkvOutput(packed, layout);
            if (!outputAdmission.admitted()) {
                throw outputAdmission.asException();
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
        AccelTensor projected = metalLinear.tryHalfLinear(input, weight, bias, config, modelPolicy, profileKey,
                outputBuffer);
        if (projected == null) {
            projected = metalLinear.tryFloatLinear(input, weight, bias, outputBuffer);
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

    LinearPair tryMetalHalfLinearPairMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer) {
        return metalLinear.tryHalfLinearPairMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer);
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
        return metalLinear.tryHalfLinearTripleMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                thirdWeight, thirdBias, profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer,
                thirdOutputBuffer);
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

    record LinearPair(AccelTensor first, AccelTensor second) {
    }

    record LinearTriple(AccelTensor first, AccelTensor second, AccelTensor third) {
    }

    record ProjectionBuffers(AccelTensor q, AccelTensor k, AccelTensor v) {
    }
}
