/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionRoutingPolicy {
    private static final String ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.allow_metal_gemma4_attention";
    private static final String DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_attention";
    private static final String ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY =
            "gollek.safetensor.allow_legacy_metal_attention_bridge";
    private static final String ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_shared_decode_packed_attention";
    private static final String DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_shared_decode_packed_attention";
    private static final String ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_paged_decode_attention";
    private static final String DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_paged_decode_attention";
    private static final String ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_raw_paged_sliding_decode_attention";
    private static final String ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_sliding_prefill_fa4_attention";
    private static final String DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_sliding_prefill_fa4_attention";
    private static final String PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_attention_max_tokens";
    private static final int DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS = 1024;
    private static final String PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_prefill_max_tokens";
    private static final int DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS = 0;
    private static final String DECODE_ATTENTION_GPU_MIN_CONTEXT_ENV =
            "GOLLEK_METAL_DECODE_ATTENTION_GPU_MIN_CONTEXT";
    private static final String DISABLE_DECODE_ATTENTION_KERNEL_ENV =
            "GOLLEK_METAL_DISABLE_DECODE_ATTENTION_KERNEL";
    private static final String FORCE_DECODE_ATTENTION_KERNEL_ENV =
            "GOLLEK_METAL_FORCE_DECODE_ATTENTION_KERNEL";
    private static final int DEFAULT_DECODE_ATTENTION_GPU_MIN_CONTEXT = 64;
    private static final String DISABLE_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_fa4_attention";
    private static final String FORCE_DENSE_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.force_dense_gemma4_attention";
    private static final boolean ALLOW_METAL_GEMMA4_ATTENTION_ENABLED =
            Boolean.getBoolean(ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY);
    private static final boolean DISABLE_METAL_GEMMA4_ATTENTION_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY);
    private static final boolean ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_ENABLED =
            Boolean.getBoolean(ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY);
    private static final boolean ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_ENABLED =
            Boolean.getBoolean(ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY);
    private static final boolean DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY);
    private static final boolean ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_ENABLED =
            Boolean.getBoolean(ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY);
    private static final boolean DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY);
    private static final boolean DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_ENABLED =
            Boolean.getBoolean(DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY);
    private static final String ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_VALUE =
            System.getProperty(ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY);
    private static final String ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_VALUE =
            System.getProperty(ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY);
    private static final boolean FORCE_DENSE_GEMMA4_ATTENTION_ENABLED =
            Boolean.getBoolean(FORCE_DENSE_GEMMA4_ATTENTION_PROPERTY);
    private static final boolean DISABLE_FA4_ATTENTION_ENABLED =
            Boolean.getBoolean(DISABLE_FA4_ATTENTION_PROPERTY);
    private static final boolean FORCE_DECODE_ATTENTION_KERNEL_ENABLED =
            envTruthy(FORCE_DECODE_ATTENTION_KERNEL_ENV);
    private static final boolean DISABLE_DECODE_ATTENTION_KERNEL_ENABLED =
            envTruthy(DISABLE_DECODE_ATTENTION_KERNEL_ENV);
    private static final int DECODE_ATTENTION_GPU_MIN_CONTEXT =
            Math.max(1, envIntOrDefault(DECODE_ATTENTION_GPU_MIN_CONTEXT_ENV,
                    DEFAULT_DECODE_ATTENTION_GPU_MIN_CONTEXT));
    private static final int PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS =
            Integer.getInteger(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY,
                    DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS);
    private static final String PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_VALUE =
            System.getProperty(PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY);
    private static final int MAX_PACKED_METAL_SLIDING_WINDOW = 2048;

    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<MetalFlashAttentionBinding> metalFa4;

    FlashAttentionRoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4) {
        this.canUseMetal = canUseMetal;
        this.metalBinding = metalBinding;
        this.metalFa4 = metalFa4;
    }

    boolean canUseMetalAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx,
            int seqLen, int startPos, float softCap) {
        if (!canUseMetal.getAsBoolean()) {
            return false;
        }
        if (!allowLegacyMetalAttentionBridge(modelPolicy) && modelPolicy.gemma4Text()) {
            return canUseGemma4SlidingPrefillFa4Attention(config, layerIdx, seqLen, startPos, softCap);
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            MetalBinding binding = metalBinding.get();
            return binding != null && binding.isWindowedAttentionAvailable();
        }
        if (modelPolicy.gemma4Text()
                && DISABLE_METAL_GEMMA4_ATTENTION_ENABLED
                && !ALLOW_METAL_GEMMA4_ATTENTION_ENABLED) {
            return false;
        }
        return true;
    }

    boolean canUseGemma4SlidingPrefillFa4Attention(ModelConfig config, int layerIdx,
            int seqLen, int startPos, float softCap) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (seqLen <= 1 || config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_ENABLED
                || DISABLE_METAL_GEMMA4_ATTENTION_ENABLED) {
            return false;
        }
        String explicit = ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_VALUE;
        if (explicit != null && !explicit.isBlank() && !Boolean.parseBoolean(explicit)) {
            return false;
        }
        int totalTokens = startPos + seqLen;
        return totalTokens <= config.slidingWindowSize() && canUseFa4Attention(softCap);
    }

    boolean canUseFa4PagedAttention(ModelConfig config, int layerIdx, float softCap) {
        if (!canUseMetal.getAsBoolean() || !canUseFa4Attention(softCap)) {
            return false;
        }
        MetalFlashAttentionBinding binding = metalFa4.get();
        if (binding == null || !binding.isNativeAvailable()) {
            return false;
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            return false;
        }
        return true;
    }

    boolean canUseSlidingDecodeMetalAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        if (!canUseMetal.getAsBoolean()) {
            return false;
        }
        if (!allowSlidingDecodeMetalAttentionBridge(config, modelPolicy, layerIdx, seqLen)) {
            return false;
        }
        if (seqLen != 1) {
            return false;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.isRuntimeActive()) {
            return false;
        }
        if (config.slidingWindowSize() > MAX_PACKED_METAL_SLIDING_WINDOW) {
            return false;
        }
        if (modelPolicy.gemma4Text()
                && DISABLE_METAL_GEMMA4_ATTENTION_ENABLED
                && !ALLOW_METAL_GEMMA4_ATTENTION_ENABLED) {
            return false;
        }
        return !config.usesSharedKvCache(layerIdx);
    }

    boolean allowSlidingDecodeMetalAttentionBridge(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (allowLegacyMetalAttentionBridge(modelPolicy)) {
            return true;
        }
        if (!modelPolicy.gemma4Text() || seqLen != 1) {
            return false;
        }
        if (config.slidingWindowSize() > MAX_PACKED_METAL_SLIDING_WINDOW) {
            return false;
        }
        return !DISABLE_METAL_GEMMA4_ATTENTION_ENABLED
                || ALLOW_METAL_GEMMA4_ATTENTION_ENABLED;
    }

    boolean canUseDenseGemma4Attention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx) {
        if (!modelPolicy.supportsForcedDenseAttention() || !FORCE_DENSE_GEMMA4_ATTENTION_ENABLED) {
            return false;
        }
        return config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow();
    }

    boolean canUseFa4Attention(float softCap) {
        if (DISABLE_FA4_ATTENTION_ENABLED) {
            return false;
        }
        MetalFlashAttentionBinding binding = metalFa4.get();
        return binding != null
                && binding.isNativeAvailable();
    }

    boolean preferPagedMetalAttentionBeforeFa4(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        if (!allowPagedMetalAttentionBridge(modelPolicy, seqLen, totalTokens)) {
            return false;
        }
        int maxTokens = PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS;
        if (maxTokens > 0 && seqLen == 1 && totalTokens <= maxTokens) {
            return true;
        }
        int prefillMaxTokens = resolvePagedMetalPrefillMaxTokens(modelPolicy);
        return prefillMaxTokens > 0
                && seqLen > 1
                && totalTokens <= prefillMaxTokens;
    }

    boolean allowPagedMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        if (allowLegacyMetalAttentionBridge(modelPolicy)) {
            return true;
        }
        if (!modelPolicy.gemma4Text() || seqLen != 1) {
            return false;
        }
        if (DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_ENABLED) {
            return false;
        }
        if (ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_ENABLED) {
            return true;
        }
        int maxTokens = PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS;
        return maxTokens > 0 && totalTokens <= maxTokens;
    }

    boolean allowLegacyMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.allowLegacyMetalAttentionBridge(ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_ENABLED);
    }

    boolean shouldUsePackedSharedDecodeAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            long seqLenQ, SharedKvState sharedKvState) {
        if (sharedKvState == null || seqLenQ != 1L) {
            return false;
        }
        if (!modelPolicy.gemma4Text()
                || DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_ENABLED
                || DISABLE_METAL_GEMMA4_ATTENTION_ENABLED) {
            return false;
        }
        String explicit = ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_VALUE;
        boolean enabled = explicit == null || explicit.isBlank() || Boolean.parseBoolean(explicit);
        MetalBinding binding = metalBinding.get();
        return enabled
                && binding != null
                && binding.isWindowedAttentionAvailable();
    }

    boolean shortDecodeUsesNativeAttention(int totalTokens) {
        if (FORCE_DECODE_ATTENTION_KERNEL_ENABLED) {
            return false;
        }
        if (DISABLE_DECODE_ATTENTION_KERNEL_ENABLED) {
            return true;
        }
        return totalTokens > 0 && totalTokens < DECODE_ATTENTION_GPU_MIN_CONTEXT;
    }

    boolean enableRawPagedSlidingDecodeAttention() {
        return ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_ENABLED;
    }

    String pagedAttentionPathName(boolean slidingLayer, long seqLen, int totalTokens, boolean firstChoice) {
        String phase;
        if (seqLen == 1) {
            phase = shortDecodeUsesNativeAttention(totalTokens) ? "native_decode" : "metal_decode";
        } else {
            phase = "native_prefill";
        }
        String window = slidingLayer ? "_windowed" : "";
        return "paged_" + phase + window + (firstChoice ? "_first" : "");
    }

    private int resolvePagedMetalPrefillMaxTokens(FlashAttentionModelPolicy modelPolicy) {
        String explicit = PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_VALUE;
        if (explicit != null && !explicit.isBlank()) {
            try {
                return Math.max(0, Integer.parseInt(explicit.trim()));
            } catch (NumberFormatException ignored) {
                return DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS;
            }
        }
        return modelPolicy.defaultPagedMetalPrefillMaxTokens(DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS);
    }

    private static int envIntOrDefault(String name, int fallback) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static boolean envTruthy(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return false;
        }
        String normalized = value.trim().toLowerCase();
        return !"0".equals(normalized)
                && !"false".equals(normalized)
                && !"no".equals(normalized)
                && !"off".equals(normalized);
    }
}
