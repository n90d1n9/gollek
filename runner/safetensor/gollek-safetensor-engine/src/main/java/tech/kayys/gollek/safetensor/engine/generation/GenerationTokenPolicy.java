/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

final class GenerationTokenPolicy {
    private static final String DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY =
            "gollek.safetensor.disable_native_greedy_argmax";
    private static final boolean DISABLE_NATIVE_GREEDY_ARGMAX_ENABLED =
            Boolean.getBoolean(DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY);
    private static final int NATIVE_ARGMAX_UNAVAILABLE = Integer.MIN_VALUE;

    private GenerationTokenPolicy() {
    }

    static boolean canUseDirectGreedySampling(GenerationConfig cfg) {
        return cfg.isPenaltyFreeGreedy();
    }

    static Set<Integer> buildBaseStopTokenIds(Tokenizer tokenizer, ModelConfig config) {
        Set<Integer> stops = new HashSet<>();
        if (tokenizer != null) {
            for (int id : tokenizer.allStopTokenIds()) {
                stops.add(id);
            }
        }
        if (config != null) {
            stops.addAll(config.eosTokenIds());
        }
        return Collections.unmodifiableSet(stops);
    }

    static Set<Integer> mergeStopTokenIds(Set<Integer> baseStops, GenerationConfig cfg) {
        List<Integer> requestStops = cfg.stopTokenIds();
        if (requestStops == null || requestStops.isEmpty()) {
            return baseStops;
        }
        Set<Integer> stops = new HashSet<>(baseStops.size() + requestStops.size());
        stops.addAll(baseStops);
        stops.addAll(requestStops);
        return stops;
    }

    static GreedySamplingMasks greedySamplingMasksFor(GreedySamplingMasks baseMasks, Tokenizer tokenizer,
            Set<Integer> stops, int vocabSize, GenerationConfig cfg, ModelRuntimeTraits traits) {
        List<Integer> requestStops = cfg.stopTokenIds();
        if (requestStops == null || requestStops.isEmpty()) {
            return baseMasks;
        }
        return buildGreedySamplingMasks(tokenizer, stops, vocabSize, traits);
    }

    static GreedySamplingMasks buildGreedySamplingMasks(Tokenizer tokenizer, Set<Integer> stops, int vocabSize) {
        return buildGreedySamplingMasks(tokenizer, stops, vocabSize, null);
    }

    static GreedySamplingMasks buildGreedySamplingMasks(Tokenizer tokenizer, Set<Integer> stops, int vocabSize,
            ModelRuntimeTraits traits) {
        return new GreedySamplingMasks(
                buildDisallowedTokenMask(tokenizer, true, stops, vocabSize, traits),
                buildDisallowedTokenMask(tokenizer, false, stops, vocabSize, traits));
    }

    static int[] initializePromptFrequencies(ModelConfig config, long[] tokenIds) {
        int[] freq = new int[config.vocabSize()];
        if (tokenIds == null || tokenIds.length == 0) {
            return freq;
        }
        for (long tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < freq.length) {
                freq[(int) tokenId]++;
            }
        }
        return freq;
    }

    static void recordSampleFrequency(int[] freq, int tokenId) {
        if (freq != null && tokenId >= 0 && tokenId < freq.length) {
            freq[tokenId]++;
        }
    }

    static int sampleNextToken(float[] logits, Tokenizer tokenizer, GenerationConfig cfg, ModelConfig config,
            ModelRuntimeTraits traits, int[] freq, Random rng, boolean firstStep, Set<Integer> stops,
            TokenSampler tokenSampler) {
        ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
        maskDisallowedContinuationTokens(logits, tokenizer, firstStep, stops, effectiveTraits);
        for (int attempt = 0; attempt < 8; attempt++) {
            int next = tokenSampler.sample(logits, cfg, config, freq, rng);
            if (!shouldRejectSampledToken(next, tokenizer, config, effectiveTraits, firstStep, stops)) {
                debugSampleChoice("accept", next, tokenizer, firstStep, attempt);
                return next;
            }
            debugSampleChoice("reject", next, tokenizer, firstStep, attempt);
            if (next >= 0 && next < logits.length) {
                logits[next] = Float.NEGATIVE_INFINITY;
            }
        }
        int fallback = tokenSampler.sample(logits, cfg, config, freq, rng);
        debugSampleChoice("fallback", fallback, tokenizer, firstStep, 8);
        return fallback;
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config) {
        try {
            return greedyArgmax(logits.dataPtr(), logits.numel(), null, 0);
        } finally {
            logits.close();
        }
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            boolean firstStep, Set<Integer> stops) {
        return sampleGreedyFromTensor(logits, config, tokenizer, null, firstStep, stops, null);
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            ModelRuntimeTraits traits, boolean firstStep, Set<Integer> stops, GreedySamplingMasks masks) {
        try {
            ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
            MemorySegment seg = logits.dataPtr();
            long vocab = logits.numel();
            Double cap = config.finalLogitSoftcapping();
            float softCap = cap != null && cap > 0 ? cap.floatValue() : 0.0f;
            BitSet baseMask = masks == null
                    ? buildDisallowedTokenMask(tokenizer, firstStep, stops, (int) vocab, effectiveTraits)
                    : masks.maskFor(firstStep);
            boolean needsTokenizerValidation = firstStep || effectiveTraits.validateContinuationTokensByDecode();
            int[] rejectedCandidates = null;
            int rejectedCount = 0;

            for (int attempt = 0; attempt < 8; attempt++) {
                if (attempt == 0) {
                    debugTopGreedyCandidates(seg, vocab, softCap, tokenizer, config, effectiveTraits, firstStep,
                            stops, 8);
                }
                int best = greedyArgmax(seg, vocab, rejectedCandidates, rejectedCount);
                if (best < 0) {
                    break;
                }
                boolean masked = baseMask != null && baseMask.get(best);
                if (!masked && (!needsTokenizerValidation
                        || !shouldRejectSampledToken(best, tokenizer, config, effectiveTraits, firstStep, stops))) {
                    debugSampleChoice("accept", best, tokenizer, firstStep, attempt);
                    return best;
                }
                debugSampleChoice("reject", best, tokenizer, firstStep, attempt);
                if (rejectedCandidates == null) {
                    rejectedCandidates = new int[8];
                }
                rejectedCandidates[rejectedCount++] = best;
            }
            return -1;
        } finally {
            logits.close();
        }
    }

    static boolean isGemma4Text(ModelConfig config) {
        return ModelRuntimeTraits.fromConfig(config).gemma4Text();
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config) {
        return encodeOptionsFor(config, null, null);
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config, String prompt) {
        return encodeOptionsFor(config, null, prompt);
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config, ModelRuntimeTraits traits, String prompt) {
        EncodeOptions options = EncodeOptions.defaultOptions();
        ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
        switch (effectiveTraits.promptBosPolicy()) {
            case NEVER -> options.addBos = false;
            case GEMMA_TURN_AWARE -> options.addBos = !looksLikeGemmaTurnPrompt(prompt);
            case DEFAULT -> {
            }
        }
        return options;
    }

    static void debugTokenSequence(Tokenizer tokenizer, long[] tokenIds, String label) {
        System.out.printf("[DEBUG-TOKENS] %s:", label);
        for (long tokenId : tokenIds) {
            String decoded = decodeSingleToken(tokenizer, (int) tokenId, false);
            String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
            System.out.printf(" [%d:%s]", tokenId, printable);
        }
        System.out.println();
        System.out.flush();
    }

    static String printableDebugText(String text) {
        if (text == null) {
            return "<null>";
        }
        return text.replace("\n", "\\n");
    }

    static void debugChosenToken(Tokenizer tokenizer, int tokenId, int step) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
        System.out.printf("[DEBUG-CHOSEN] step=%d token=%d text=%s%n", step, tokenId, printable);
        System.out.flush();
    }

    private static int greedyArgmax(MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        int nativeBest = tryNativeGreedyArgmax(seg, vocab, rejectedCandidates, rejectedCount);
        if (nativeBest != NATIVE_ARGMAX_UNAVAILABLE) {
            return nativeBest;
        }
        return javaGreedyArgmax(seg, vocab, rejectedCandidates, rejectedCount);
    }

    private static int tryNativeGreedyArgmax(MemorySegment seg,
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

    private static int javaGreedyArgmax(MemorySegment seg,
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

    private static void maskDisallowedContinuationTokens(float[] logits, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, ModelRuntimeTraits traits) {
        if (logits == null || tokenizer == null) {
            return;
        }
        BitSet mask = buildDisallowedTokenMask(tokenizer, firstStep, stops, logits.length, traits);
        for (int tokenId = mask.nextSetBit(0); tokenId >= 0; tokenId = mask.nextSetBit(tokenId + 1)) {
            logits[tokenId] = Float.NEGATIVE_INFINITY;
        }
    }

    private static BitSet buildDisallowedTokenMask(Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, int vocabSize, ModelRuntimeTraits traits) {
        BitSet mask = new BitSet(Math.max(0, vocabSize));
        if (tokenizer == null || vocabSize <= 0) {
            return mask;
        }
        setIfInVocab(mask, tokenizer.bosTokenId(), vocabSize);
        setIfInVocab(mask, tokenizer.padTokenId(), vocabSize);

        Map<String, Integer> specialTokens = tokenizer.specialTokens();
        if (specialTokens != null && !specialTokens.isEmpty()) {
            for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
                Integer id = entry.getValue();
                if (id == null || id < 0 || id >= vocabSize) {
                    continue;
                }
                String text = entry.getKey();
                if (text != null && isAllowedControlText(text.trim(), traits)) {
                    continue;
                }
                if (stops == null || !stops.contains(id)) {
                    mask.set(id);
                }
            }
        }

        if (firstStep && stops != null) {
            for (Integer stop : stops) {
                if (stop != null) {
                    setIfInVocab(mask, stop, vocabSize);
                }
            }
        }
        return mask;
    }

    private static void setIfInVocab(BitSet mask, int tokenId, int vocabSize) {
        if (tokenId >= 0 && tokenId < vocabSize) {
            mask.set(tokenId);
        }
    }

    private static boolean isDisallowedContinuationToken(int tokenId, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, ModelRuntimeTraits traits) {
        if (tokenizer == null || tokenId < 0) {
            return false;
        }
        if (tokenId == tokenizer.bosTokenId() || tokenId == tokenizer.padTokenId()) {
            return true;
        }
        String specialText = specialTokenText(tokenizer, tokenId);
        if (specialText != null) {
            if (isAllowedControlText(specialText.trim(), traits)) {
                return false;
            }
            return stops == null || !stops.contains(tokenId);
        }
        return firstStep && stops != null && stops.contains(tokenId);
    }

    private static boolean shouldRejectSampledToken(int tokenId, Tokenizer tokenizer, ModelConfig config,
            ModelRuntimeTraits traits, boolean firstStep, Set<Integer> stops) {
        if (tokenId < 0) {
            return false;
        }
        if (isDisallowedContinuationToken(tokenId, tokenizer, firstStep, stops, traits)) {
            return true;
        }
        if (!firstStep && !traits.validateContinuationTokensByDecode()) {
            return false;
        }
        if (tokenizer == null) {
            return false;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        if (decoded == null) {
            return false;
        }
        String trimmed = decoded.trim();
        if (isAllowedControlText(trimmed, traits)) {
            return false;
        }
        if (firstStep && traits.gemma4Text()
                && ("model".equalsIgnoreCase(trimmed) || "assistant".equalsIgnoreCase(trimmed))) {
            return true;
        }
        // Reject only truly empty decodes, not whitespace (space/newline) — those are
        // valid first tokens.
        if (traits.rejectEmptyDecodedTokens() && decoded.isEmpty()) {
            return true;
        }
        if (!firstStep) {
            return false;
        }
        return trimmed.startsWith("<|")
                || trimmed.endsWith("|>")
                || (trimmed.startsWith("<") && trimmed.endsWith(">"))
                || trimmed.startsWith("<unused");
    }

    private static String specialTokenText(Tokenizer tokenizer, int tokenId) {
        if (tokenizer == null || tokenId < 0) {
            return null;
        }
        Map<String, Integer> specialTokens = tokenizer.specialTokens();
        if (specialTokens == null || specialTokens.isEmpty()) {
            return null;
        }
        for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
            Integer id = entry.getValue();
            if (id != null && id == tokenId) {
                return entry.getKey();
            }
        }
        return null;
    }

    private static boolean isAllowedControlText(String text, ModelRuntimeTraits traits) {
        return text != null && traits.allowedControlTokenTexts().contains(text);
    }

    private static String decodeSingleToken(Tokenizer tokenizer, int tokenId, boolean skipSpecialTokens) {
        try {
            DecodeOptions options = DecodeOptions.defaultOptions().skipSpecialTokens(skipSpecialTokens);
            return tokenizer.decode(new long[] { tokenId }, options);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void debugSampleChoice(String decision, int tokenId, Tokenizer tokenizer, boolean firstStep,
            int attempt) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
        System.out.printf("[DEBUG-SAMPLE] %s firstStep=%s attempt=%d token=%d text=%s%n",
                decision, firstStep, attempt, tokenId, printable);
        System.out.flush();
    }

    private static void debugTopGreedyCandidates(MemorySegment seg,
            long vocab,
            float softCap,
            Tokenizer tokenizer,
            ModelConfig config,
            ModelRuntimeTraits traits,
            boolean firstStep,
            Set<Integer> stops,
            int limit) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        int topN = Math.max(1, limit);
        int[] ids = new int[topN];
        float[] vals = new float[topN];
        Arrays.fill(ids, -1);
        Arrays.fill(vals, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < vocab; i++) {
            float value = applyLogitSoftcap(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i), softCap);
            for (int slot = 0; slot < topN; slot++) {
                if (value > vals[slot]) {
                    for (int shift = topN - 1; shift > slot; shift--) {
                        vals[shift] = vals[shift - 1];
                        ids[shift] = ids[shift - 1];
                    }
                    vals[slot] = value;
                    ids[slot] = i;
                    break;
                }
            }
        }

        System.out.printf("[DEBUG-TOP] firstStep=%s%n", firstStep);
        for (int rank = 0; rank < topN; rank++) {
            int tokenId = ids[rank];
            if (tokenId < 0) {
                continue;
            }
            boolean disallowed = isDisallowedContinuationToken(tokenId, tokenizer, firstStep, stops, traits);
            boolean rejected = shouldRejectSampledToken(tokenId, tokenizer, config, traits, firstStep, stops);
            String decoded = decodeSingleToken(tokenizer, tokenId, false);
            String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
            System.out.printf("[DEBUG-TOP] rank=%d token=%d logit=%f disallowed=%s rejected=%s text=%s%n",
                    rank + 1, tokenId, vals[rank], disallowed, rejected, printable);
        }
        System.out.flush();
    }

    private static float applyLogitSoftcap(float value, float softCap) {
        if (softCap <= 0.0f) {
            return value;
        }
        return (float) (softCap * Math.tanh(value / softCap));
    }

    private static ModelRuntimeTraits effectiveTraits(ModelConfig config, ModelRuntimeTraits traits) {
        return traits == null ? ModelRuntimeTraits.fromConfig(config) : traits;
    }

    private static boolean looksLikeGemmaTurnPrompt(String prompt) {
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.stripLeading();
        if (trimmed.startsWith("<bos>")) {
            trimmed = trimmed.substring("<bos>".length()).stripLeading();
        }
        return trimmed.startsWith("<start_of_turn>")
                || trimmed.startsWith("<|turn>");
    }
}
