/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardNativeBf16MatvecPolicy.describeNativeBf16MatvecPath;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalHalfLinearExecutionPlan(
        int rows,
        int inputDim,
        int outputDim,
        boolean nativeBf16Weight,
        boolean nativeBf16MatvecCandidate,
        boolean mpsHalfMatvecCandidate,
        boolean transposedHalfMatvecCandidate,
        boolean halfMatvecCandidate) {

    static DirectForwardMetalHalfLinearExecutionPlan from(
            int rows,
            int inputDim,
            int outputDim,
            boolean nativeBf16Weight,
            DirectForwardMetalCapabilities capabilities,
            boolean halfMatvecAllowed,
            boolean mpsHalfMatvecAllowed,
            boolean transposedHalfMatvecAllowed) {
        boolean singleRow = rows == 1;
        return new DirectForwardMetalHalfLinearExecutionPlan(
                rows,
                inputDim,
                outputDim,
                nativeBf16Weight,
                singleRow
                        && nativeBf16Weight
                        && halfMatvecAllowed
                        && capabilities.supportsMatvecTransposedRightBf16(),
                singleRow
                        && !nativeBf16Weight
                        && mpsHalfMatvecAllowed
                        && capabilities.supportsMatvecTransposedRightHalfMps(),
                singleRow
                        && !nativeBf16Weight
                        && transposedHalfMatvecAllowed
                        && capabilities.supportsMatvecTransposedWeightHalf(),
                singleRow
                        && !nativeBf16Weight
                        && halfMatvecAllowed
                        && capabilities.supportsMatvecTransposedRightHalf());
    }

    String nativeBf16MatvecPath() {
        return describeNativeBf16MatvecPath(inputDim, outputDim);
    }

    String mpsHalfMatvecPath() {
        return "mps_matvec";
    }

    String transposedHalfMatvecPath() {
        return "transposed_matvec";
    }

    String halfMatvecPath() {
        return "matvec";
    }

    String matmulPath() {
        return "metal_matmul";
    }

    boolean matchesTransposedWeight(AccelTensor transposedWeight) {
        return transposedWeight != null
                && transposedWeight.size(0) == inputDim
                && transposedWeight.size(1) == outputDim;
    }
}
