/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.metal.binding.MetalBinding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class GenerationGreedyArgmax {
    private static final String DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY =
            "gollek.safetensor.disable_native_greedy_argmax";
    private static final boolean DISABLE_NATIVE_GREEDY_ARGMAX_ENABLED =
            Boolean.getBoolean(DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY);
    private static final int NATIVE_ARGMAX_UNAVAILABLE = Integer.MIN_VALUE;

    private GenerationGreedyArgmax() {
    }

    static int select(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        int nativeBest = tryNative(seg, vocab, rejectedCandidates, rejectedCount);
        if (nativeBest != NATIVE_ARGMAX_UNAVAILABLE) {
            return nativeBest;
        }
        return selectJava(seg, vocab, rejectedCandidates, rejectedCount);
    }

    static int selectJava(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            if (rejectedCount > 0 && containsRejectedCandidate(rejectedCandidates, rejectedCount, i)) {
                continue;
            }
            // Logit softcap is monotonic, so it cannot change greedy argmax ordering.
            float value = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    static int selectJava(float[] logits) {
        return selectJava(logits, logits.length, null, 0);
    }

    static int selectJava(
            float[] logits,
            int vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        int limit = Math.min(vocab, logits.length);
        for (int i = 0; i < limit; i++) {
            if (rejectedCount > 0 && containsRejectedCandidate(rejectedCandidates, rejectedCount, i)) {
                continue;
            }
            float value = logits[i];
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    private static int tryNative(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        if (DISABLE_NATIVE_GREEDY_ARGMAX_ENABLED || vocab > Integer.MAX_VALUE) {
            return NATIVE_ARGMAX_UNAVAILABLE;
        }
        try {
            MetalBinding binding = MetalBinding.getInstance();
            if (binding == null || !binding.supportsArgmaxF32()) {
                return NATIVE_ARGMAX_UNAVAILABLE;
            }
            return binding.argmaxF32(
                    seg,
                    (int) vocab,
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 0),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 1),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 2),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 3),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 4),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 5),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 6),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 7));
        } catch (Throwable ignored) {
            return NATIVE_ARGMAX_UNAVAILABLE;
        }
    }

    private static int rejectedCandidateAt(int[] rejectedCandidates, int rejectedCount, int index) {
        if (rejectedCandidates == null || index >= rejectedCount) {
            return -1;
        }
        return rejectedCandidates[index];
    }

    private static boolean containsRejectedCandidate(int[] rejectedCandidates, int rejectedCount, int tokenId) {
        if (rejectedCandidates == null) {
            return false;
        }
        for (int i = 0; i < rejectedCount; i++) {
            if (rejectedCandidates[i] == tokenId) {
                return true;
            }
        }
        return false;
    }
}
