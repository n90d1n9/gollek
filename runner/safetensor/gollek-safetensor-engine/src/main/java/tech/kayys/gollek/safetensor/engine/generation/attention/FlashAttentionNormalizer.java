/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

final class FlashAttentionNormalizer {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final String DISABLE_GEMMA4_V_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_v_norm";
    private static final String DISABLE_GEMMA4_QK_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_qk_norm";
    private static final String ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_per_head_rms_norm";
    private static final String DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_per_head_rms_norm";
    private static final boolean DISABLE_METAL_PER_HEAD_RMS_NORM_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY);
    private static final String ENABLE_METAL_PER_HEAD_RMS_NORM_VALUE =
            System.getProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY);

    private final Supplier<MetalBinding> metalBinding;
    private final Map<Integer, AccelTensor> unitScaleRmsNormWeights = new ConcurrentHashMap<>();

    FlashAttentionNormalizer(Supplier<MetalBinding> metalBinding) {
        this.metalBinding = metalBinding;
    }

    boolean gemma4QkNormDisabled(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.gemma4Text() && Boolean.getBoolean(DISABLE_GEMMA4_QK_NORM_PROPERTY);
    }

    boolean gemma4VNormDisabled(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.gemma4Text() && Boolean.getBoolean(DISABLE_GEMMA4_V_NORM_PROPERTY);
    }

    AccelTensor perHeadRmsNorm(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy) {
        AccelTensor normed = tryMetalPerHeadRmsNorm(x, weight, eps, addOne, modelPolicy);
        if (normed != null) {
            return normed;
        }
        return AccelOps.perHeadRmsNorm(x, weight, eps, addOne);
    }

    AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps, FlashAttentionModelPolicy modelPolicy) {
        AccelTensor normed = tryMetalPerHeadRmsNormNoWeight(x, eps, modelPolicy);
        if (normed != null) {
            return normed;
        }
        return perHeadRmsNormNoWeight(x, eps);
    }

    AccelTensor rmsNorm(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        return AccelOps.rmsNorm(x, w, eps, addOne);
    }

    private AccelTensor tryMetalPerHeadRmsNorm(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy) {
        if (!shouldUseMetalPerHeadRmsNorm(modelPolicy) || x == null || weight == null) {
            return null;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.nativeElementwiseKernelsAvailable()) {
            return null;
        }
        if (x.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        long[] shape = x.shape();
        if (shape.length < 2) {
            return null;
        }
        int headDim = Math.toIntExact(shape[shape.length - 1]);
        if (headDim <= 0 || weight.numel() != headDim) {
            return null;
        }

        AccelTensor weightView = weight;
        AccelTensor contiguousInput = null;
        AccelTensor out = null;
        try {
            if (weight.quantType() != AccelTensor.QuantType.F32) {
                weightView = weight.dequantize();
            }
            if (weightView.quantType() != AccelTensor.QuantType.F32 || weightView.numel() != headDim) {
                return null;
            }
            contiguousInput = x.contiguous();
            out = AccelTensor.zeros(shape);
            int rows = Math.toIntExact(contiguousInput.numel() / headDim);
            int rc = binding.rmsNormRows(out.dataPtr(), contiguousInput.dataPtr(), weightView.dataPtr(),
                    rows, headDim, (float) eps, addOne);
            if (rc == 0) {
                return out;
            }
            out.close();
            out = null;
            return null;
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (contiguousInput != null && contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
            if (weightView != weight && !weightView.isClosed()) {
                weightView.close();
            }
        }
    }

    private AccelTensor tryMetalPerHeadRmsNormNoWeight(AccelTensor x, double eps,
            FlashAttentionModelPolicy modelPolicy) {
        if (!shouldUseMetalPerHeadRmsNorm(modelPolicy) || x == null) {
            return null;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.nativeElementwiseKernelsAvailable()) {
            return null;
        }
        if (x.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        long[] shape = x.shape();
        if (shape.length < 2) {
            return null;
        }
        int headDim = Math.toIntExact(shape[shape.length - 1]);
        if (headDim <= 0) {
            return null;
        }

        AccelTensor contiguousInput = null;
        AccelTensor out = null;
        try {
            AccelTensor zeroWeight = unitScaleRmsNormWeights.computeIfAbsent(headDim, dim -> AccelTensor.zeros(dim));
            contiguousInput = x.contiguous();
            out = AccelTensor.zeros(shape);
            int rows = Math.toIntExact(contiguousInput.numel() / headDim);
            int rc = binding.rmsNormRows(out.dataPtr(), contiguousInput.dataPtr(), zeroWeight.dataPtr(),
                    rows, headDim, (float) eps, true);
            if (rc == 0) {
                return out;
            }
            out.close();
            out = null;
            return null;
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (contiguousInput != null && contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private boolean shouldUseMetalPerHeadRmsNorm(FlashAttentionModelPolicy modelPolicy) {
        if (DISABLE_METAL_PER_HEAD_RMS_NORM_ENABLED) {
            return false;
        }
        String explicit = ENABLE_METAL_PER_HEAD_RMS_NORM_VALUE;
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return modelPolicy.preferMetalPerHeadRmsNorm();
    }

    private AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps) {
        AccelTensor contiguousInput = x.contiguous();
        try {
            long[] shape = contiguousInput.shape();
            int headDim = (int) shape[shape.length - 1];
            int numHeads = (int) shape[shape.length - 2];
            int outer = (int) (contiguousInput.numel() / (numHeads * headDim));

            AccelTensor out = AccelTensor.zeros(shape);
            MemorySegment xSeg = contiguousInput.dataSegment();
            MemorySegment oSeg = out.dataSegment();

            for (int b = 0; b < outer; b++) {
                for (int h = 0; h < numHeads; h++) {
                    long base = (long) (b * numHeads + h) * headDim;
                    int upperBound = FLOAT_SPECIES.loopBound(headDim);
                    FloatVector sumSqVec = FloatVector.zero(FLOAT_SPECIES);
                    int i = 0;
                    for (; i < upperBound; i += FLOAT_SPECIES.length()) {
                        FloatVector valueVec = FloatVector.fromMemorySegment(
                                FLOAT_SPECIES, xSeg, (base + i) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        sumSqVec = valueVec.fma(valueVec, sumSqVec);
                    }
                    float sumSq = sumSqVec.reduceLanes(VectorOperators.ADD);
                    for (; i < headDim; i++) {
                        float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                        sumSq += val * val;
                    }
                    float rms = (float) (1.0 / Math.sqrt(sumSq / headDim + eps));
                    FloatVector rmsVec = FloatVector.broadcast(FLOAT_SPECIES, rms);
                    int j = 0;
                    for (; j < upperBound; j += FLOAT_SPECIES.length()) {
                        FloatVector valueVec = FloatVector.fromMemorySegment(
                                FLOAT_SPECIES, xSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        valueVec.mul(rmsVec).intoMemorySegment(
                                oSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                    }
                    for (; j < headDim; j++) {
                        float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + j);
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + j, val * rms);
                    }
                }
            }
            return out;
        } finally {
            if (contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }
}
