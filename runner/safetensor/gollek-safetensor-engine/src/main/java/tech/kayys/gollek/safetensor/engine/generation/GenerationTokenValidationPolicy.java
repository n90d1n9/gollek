/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.BitSet;
import java.util.Map;
import java.util.Set;

final class GenerationTokenValidationPolicy {
    private GenerationTokenValidationPolicy() {
    }

    static GreedySamplingMasks buildGreedySamplingMasks(Tokenizer tokenizer, Set<Integer> stops, int vocabSize,
            ModelRuntimeTraits traits) {
        return new GreedySamplingMasks(
                buildDisallowedTokenMask(tokenizer, true, stops, vocabSize, traits),
                buildDisallowedTokenMask(tokenizer, false, stops, vocabSize, traits));
    }

    static void maskDisallowedContinuationTokens(float[] logits, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, ModelRuntimeTraits traits) {
        if (logits == null || tokenizer == null) {
            return;
        }
        BitSet mask = buildDisallowedTokenMask(tokenizer, firstStep, stops, logits.length, traits);
        for (int tokenId = mask.nextSetBit(0); tokenId >= 0; tokenId = mask.nextSetBit(tokenId + 1)) {
            logits[tokenId] = Float.NEGATIVE_INFINITY;
        }
    }

    static BitSet buildDisallowedTokenMask(Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, int vocabSize, ModelRuntimeTraits traits) {
        ModelRuntimeTraits effectiveTraits = traits == null ? ModelRuntimeTraits.EMPTY : traits;
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
                if (text != null && isAllowedControlText(text.trim(), effectiveTraits)) {
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

    static boolean isDisallowedContinuationToken(int tokenId, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, ModelRuntimeTraits traits) {
        ModelRuntimeTraits effectiveTraits = traits == null ? ModelRuntimeTraits.EMPTY : traits;
        if (tokenizer == null || tokenId < 0) {
            return false;
        }
        if (tokenId == tokenizer.bosTokenId() || tokenId == tokenizer.padTokenId()) {
            return true;
        }
        String specialText = specialTokenText(tokenizer, tokenId);
        if (specialText != null) {
            if (isAllowedControlText(specialText.trim(), effectiveTraits)) {
                return false;
            }
            return stops == null || !stops.contains(tokenId);
        }
        return firstStep && stops != null && stops.contains(tokenId);
    }

    static boolean shouldRejectSampledToken(int tokenId, Tokenizer tokenizer, ModelRuntimeTraits traits,
            boolean firstStep, Set<Integer> stops) {
        ModelRuntimeTraits effectiveTraits = traits == null ? ModelRuntimeTraits.EMPTY : traits;
        if (tokenId < 0) {
            return false;
        }
        if (isDisallowedContinuationToken(tokenId, tokenizer, firstStep, stops, effectiveTraits)) {
            return true;
        }
        if (!firstStep && !effectiveTraits.validateContinuationTokensByDecode()) {
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
        if (isAllowedControlText(trimmed, effectiveTraits)) {
            return false;
        }
        if (firstStep && effectiveTraits.gemma4Text()
                && ("model".equalsIgnoreCase(trimmed) || "assistant".equalsIgnoreCase(trimmed))) {
            return true;
        }
        // Reject only truly empty decodes, not whitespace (space/newline) — those are
        // valid first tokens.
        if (effectiveTraits.rejectEmptyDecodedTokens() && decoded.isEmpty()) {
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

    static String decodeSingleToken(Tokenizer tokenizer, int tokenId, boolean skipSpecialTokens) {
        try {
            DecodeOptions options = DecodeOptions.defaultOptions().skipSpecialTokens(skipSpecialTokens);
            return tokenizer.decode(new long[] { tokenId }, options);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void setIfInVocab(BitSet mask, int tokenId, int vocabSize) {
        if (tokenId >= 0 && tokenId < vocabSize) {
            mask.set(tokenId);
        }
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
}
