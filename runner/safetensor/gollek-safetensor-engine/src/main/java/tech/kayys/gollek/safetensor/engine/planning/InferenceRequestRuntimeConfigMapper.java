/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendCapabilities;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.util.Locale;

/**
 * Maps provider-level request knobs into runtime configs consumed by safetensor execution.
 */
@ApplicationScoped
public class InferenceRequestRuntimeConfigMapper {
    private static final Logger log = Logger.getLogger(InferenceRequestRuntimeConfigMapper.class);
    private static final int DEFAULT_MAX_TOKENS = 256;

    GenerationConfig generationConfig(
            ProviderRequest request,
            TextExecutionBackendCapabilities capabilities) {
        int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : DEFAULT_MAX_TOKENS;
        int topK = request.getTopK();
        float topP = (float) request.getTopP();
        float temperature = (float) request.getTemperature();
        float minP = request.getParameter("min_p", Number.class)
                .map(Number::floatValue)
                .orElse(0.0f);
        long seed = request.getParameter("seed", Number.class)
                .map(Number::longValue)
                .orElse(-1L);
        String kvQuantStr = request.getParameter("kv_cache_quant", String.class).orElse("none");
        GenerationConfig.KvCacheQuantization kvQuant = normalizeKvQuantization(kvQuantStr, capabilities);

        return GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveSamplingStrategy(temperature, topK, topP))
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .minP(minP)
                .repetitionPenalty((float) request.getRepeatPenalty())
                .kvCacheQuant(kvQuant)
                .seed(seed)
                .build();
    }

    AudioConfig audioConfig(ProviderRequest request) {
        String outputFormat = request.getParameter("output_format", String.class).orElse("wav");
        return AudioConfig.builder()
                .temperature((float) request.getTemperature())
                .format(resolveAudioFormat(outputFormat))
                .build();
    }

    static GenerationConfig.KvCacheQuantization normalizeKvQuantization(
            String raw,
            TextExecutionBackendCapabilities capabilities) {
        if (raw == null || raw.isBlank() || "none".equalsIgnoreCase(raw)) {
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        if (capabilities == null || !capabilities.supportsKvCacheQuantization()) {
            log.warnf("KV cache quantization '%s' is not supported by backend '%s'; using NONE", raw, capabilities);
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        String normalized = raw.trim().toUpperCase(Locale.ROOT);
        if ("INT8".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT8;
        }
        if ("INT4".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        if ("TURBO".equals(normalized)) {
            log.warnf("KV cache quantization '%s' falls back to packed INT4 in the direct safetensor runtime", raw);
            return GenerationConfig.KvCacheQuantization.INT4;
        }
        return GenerationConfig.KvCacheQuantization.NONE;
    }

    static GenerationConfig.SamplingStrategy resolveSamplingStrategy(float temperature, int topK, float topP) {
        if (temperature < 1.0e-4f || topK == 1) {
            return GenerationConfig.SamplingStrategy.GREEDY;
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0f && topP < 1.0f;
        if (hasTopK && hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_K_TOP_P;
        }
        if (hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_P;
        }
        if (hasTopK) {
            return GenerationConfig.SamplingStrategy.TOP_K;
        }
        return GenerationConfig.SamplingStrategy.GREEDY;
    }

    private static AudioConfig.Format resolveAudioFormat(String raw) {
        if (raw == null || raw.isBlank()) {
            return AudioConfig.Format.WAV;
        }
        try {
            return AudioConfig.Format.valueOf(raw.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException ignored) {
            return AudioConfig.Format.WAV;
        }
    }
}
