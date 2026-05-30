/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

final class DirectForwardMetalFloatLinear {
    private DirectForwardMetalFloatLinear() {
    }

    static AccelTensor tryLinear(Logger log,
                                 MetalBinding metalBinding,
                                 boolean metalLinearEnabled,
                                 AccelTensor input,
                                 AccelTensor weight,
                                 AccelTensor bias) {
        if (!canUseCandidate(metalLinearEnabled, metalBinding, input, weight)) {
            return null;
        }
        AccelTensor contiguousInput = input.contiguous();
        long outputDim = weight.size(0);
        long[] outputShape = input.shapeWithLastDim(outputDim);
        AccelTensor out = AccelTensor.zeros(outputShape);

        try {
            long k = input.size(-1);
            long rows = input.numel() / Math.max(1L, k);
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(outputDim);
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
            log.debugf("Falling back from Metal float linear to AccelOps: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static boolean canUseCandidate(boolean metalLinearEnabled,
                                           MetalBinding metalBinding,
                                           AccelTensor input,
                                           AccelTensor weight) {
        if (!metalLinearEnabled || metalBinding == null) {
            return false;
        }
        if (input == null || weight == null) {
            return false;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        int inputRank = input.rank();
        if (inputRank < 2 || weight.rank() != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, input.size(-1));
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputRank - 2; i++) {
            batchProduct *= input.size(i);
        }
        return batchProduct == 1L;
    }
}
