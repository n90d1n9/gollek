/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Arrays;
import java.util.Map;

/**
 * Immutable continuation plan for conversation KV-cache reuse.
 */
record DirectConversationContinuationPlan(
        long[] fullInputIds,
        int cachedPrefixTokens,
        long[] deltaInputIds,
        boolean exactReplay,
        int replayTokenId) {

    private static final int NO_REPLAY_TOKEN = -1;

    static DirectConversationContinuationPlan resolve(
            long[] fullInputIds,
            int cachedPrefixTokens,
            int currentCachePosition,
            Integer replayTokenId) {
        if (fullInputIds == null) {
            throw new IllegalArgumentException("Conversation continuation requires input tokens");
        }
        if (cachedPrefixTokens < 0 || cachedPrefixTokens > fullInputIds.length) {
            throw new IllegalArgumentException("Invalid cachedPrefixTokens: " + cachedPrefixTokens);
        }
        if (currentCachePosition != cachedPrefixTokens) {
            throw new IllegalStateException("KV cache position " + currentCachePosition
                    + " does not match cached prefix tokens " + cachedPrefixTokens);
        }

        long[] deltaInputIds = Arrays.copyOfRange(fullInputIds, cachedPrefixTokens, fullInputIds.length);
        boolean exactReplay = deltaInputIds.length == 0;
        if (exactReplay && replayTokenId == null) {
            throw new IllegalArgumentException(
                    "Conversation continuation requires at least one delta token or a replay token");
        }

        return new DirectConversationContinuationPlan(
                fullInputIds.clone(),
                cachedPrefixTokens,
                deltaInputIds,
                exactReplay,
                replayTokenId == null ? NO_REPLAY_TOKEN : replayTokenId);
    }

    int nextReplayTokenId() {
        if (!exactReplay) {
            throw new IllegalStateException("Continuation plan is not an exact replay");
        }
        return replayTokenId;
    }

    String promptTokenSource() {
        return exactReplay ? "conversation_replay" : "conversation_delta";
    }

    Map<String, Object> retainedKvMetadata() {
        return Map.of(
                "prompt_token_source", promptTokenSource(),
                "conversation_delta_prefill", !exactReplay,
                "conversation_exact_replay", exactReplay,
                "conversation_cached_prefix_tokens", cachedPrefixTokens,
                "conversation_delta_prompt_tokens", deltaInputIds.length,
                "conversation_kv_retained", true);
    }
}
