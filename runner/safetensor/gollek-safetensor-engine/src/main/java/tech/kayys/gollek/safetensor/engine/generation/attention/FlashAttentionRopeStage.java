/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

final class FlashAttentionRopeStage {
    private static final String EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY =
            "gollek.safetensor.experimental_gemma4_split_half_rope";
    private static final String LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY =
            "gollek.safetensor.legacy_interleaved_gemma4_rope";

    private final RopeFrequencyCache ropeCache;

    FlashAttentionRopeStage(RopeFrequencyCache ropeCache) {
        this.ropeCache = ropeCache;
    }

    void apply(AccelTensor q, AccelTensor k, int startPos, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy, int layerIdx, int headDim) {
        // Hugging Face Gemma-4 text uses split-half `rotate_half`, not the
        // legacy adjacent-pair interleaved rotation. Keep the old path only
        // as an explicit escape hatch while we finish the remaining parity
        // work around shared-KV and full/sliding attention interaction.
        boolean gemma4LegacyInterleavedRope = modelPolicy.gemma4Text()
                && Boolean.getBoolean(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY);
        boolean interleavedRope = modelPolicy.useInterleavedRope(
                gemma4LegacyInterleavedRope, Boolean.getBoolean(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY));
        int rotatedDim = resolveRotatedDim(config, layerIdx, headDim);
        int rotaryDim = resolveRotaryStorageDim(headDim, rotatedDim, interleavedRope, modelPolicy);
        RopeFrequencyCache.RopeFrequencies freqs = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(),
                config.ropeThetaForLayer(layerIdx), config.ropeScaling(),
                resolveRopeExponentDenominator(rotaryDim), Math.min(rotaryDim, rotatedDim));
        applyRope(q, k, startPos, freqs, interleavedRope);
    }

    private void applyRope(AccelTensor q, AccelTensor k, int startPos, RopeFrequencyCache.RopeFrequencies freqs,
            boolean interleaved) {
        int seqLen = (int) q.size(1);
        int numQHeads = (int) q.size(2);
        int headDim = (int) q.size(3);

        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numQHeads; h++) {
                freqs.rotateInPlace(q.dataPtr(), ((long) s * numQHeads + h) * headDim, pos, interleaved);
            }
            if (k != null) {
                int numKVHeads = (int) k.size(2);
                for (int h = 0; h < numKVHeads; h++) {
                    freqs.rotateInPlace(k.dataPtr(), ((long) s * numKVHeads + h) * headDim, pos, interleaved);
                }
            }
        }
    }

    private int resolveRotaryStorageDim(int headDim, int rotatedDim, boolean interleaved,
            FlashAttentionModelPolicy modelPolicy) {
        // Partial RoPE (Gemma-4 proportional full attention) still uses the
        // full head layout. Only the first rotated span receives non-identity
        // frequencies; the remaining split-half pairs must stay in place.
        if (!interleaved && !modelPolicy.gemma4Text() && rotatedDim < headDim) {
            return rotatedDim;
        }
        return headDim;
    }

    private int resolveRotatedDim(ModelConfig config, int layerIdx, int storageDim) {
        double partialFactor = config.partialRotaryFactorForLayer(layerIdx);
        int rotaryDim = (int) Math.round(storageDim * partialFactor);
        rotaryDim = Math.max(2, rotaryDim);
        if ((rotaryDim & 1) != 0) {
            rotaryDim--;
        }
        return Math.min(storageDim, rotaryDim);
    }

    private int resolveRopeExponentDenominator(int rotaryDim) {
        return rotaryDim;
    }
}
