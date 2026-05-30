/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

final class InferenceProfile {
    final String mode;
    final boolean detailed;
    final HostLoadSnapshot hostLoadStart;
    HostLoadSnapshot hostLoadEnd;
    long tokenizeNanos;
    long modelLoadNanos;
    long sessionAllocateNanos;
    long firstTokenNanos;
    long prefillNanos;
    long decodeNanos;
    long samplingNanos;
    long attentionNanos;
    long ffnNanos;
    long logitsProjectionNanos;
    long logitsMaterializationNanos;
    final Map<String, Long> linearNanosByOperation = new LinkedHashMap<>();
    final Map<String, Integer> linearPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> ffnPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> attentionPathCounts = new LinkedHashMap<>();
    int decodeSteps;

    InferenceProfile(String mode, boolean detailed) {
        this.mode = mode;
        this.detailed = detailed;
        this.hostLoadStart = HostLoadSnapshot.capture();
    }

    Map<String, Object> metadata(String backend) {
        return metadata(backend, -1, -1);
    }

    /**
     * @param promptTokens     prompt token count for llama.cpp-style bench keys;
     *                         {@code -1} skips bench/tokens
     * @param completionTokens generated token count (excluding EOS stop)
     */
    Map<String, Object> metadata(String backend, int promptTokens, int completionTokens) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("profile_mode", mode);
        metadata.put("profile_backend", backend);
        metadata.put("profile_load_ms", roundMillis(modelLoadNanos));
        if (modelLoadNanos > 0L) {
            metadata.put("bench.load_ms", roundMillis(modelLoadNanos));
        }
        metadata.put("profile_tokenize_ms", roundMillis(tokenizeNanos));
        metadata.put("profile_session_allocate_ms", roundMillis(sessionAllocateNanos));
        if (firstTokenNanos > 0L) {
            metadata.put("profile_ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("bench.ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("bench.cold_ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("profile_engine_ttft_ms", roundMillis(engineFirstTokenNanos()));
            metadata.put("bench.engine_ttft_ms", roundMillis(engineFirstTokenNanos()));
            metadata.put("profile_engine_ttft_excluded_ms",
                    roundMillis(engineFirstTokenExcludedNanos()));
        }
        metadata.put("profile_prefill_ms", roundMillis(prefillNanos));
        metadata.put("profile_decode_ms", roundMillis(decodeNanos));
        metadata.put("profile_sampling_ms", roundMillis(samplingNanos));
        metadata.put("profile_attention_ms", roundMillis(attentionNanos));
        metadata.put("profile_ffn_ms", roundMillis(ffnNanos));
        metadata.put("profile_logits_ms", roundMillis(logitsProjectionNanos));
        metadata.put("profile_logits_materialization_ms", roundMillis(logitsMaterializationNanos));
        putHostLoadMetadata(metadata, hostLoadStart, hostLoadEnd());
        linearNanosByOperation.forEach((operation, nanos) -> metadata
                .put("profile_linear_" + sanitizeMetricKey(operation) + "_ms", roundMillis(nanos)));
        linearPathCounts.forEach((path, count) -> metadata
                .put("profile_linear_path_" + sanitizeMetricKey(path) + "_count", count));
        ffnPathCounts.forEach((path, count) -> metadata
                .put("profile_ffn_path_" + sanitizeMetricKey(path) + "_count", count));
        attentionPathCounts.forEach((path, count) -> metadata
                .put("profile_attention_path_" + sanitizeMetricKey(path) + "_count", count));
        metadata.put("profile_decode_steps", decodeSteps);
        metadata.put("profile_summary", summary(backend));
        if (promptTokens >= 0) {
            metadata.put("tokens.input", promptTokens);
            metadata.put("tokens.output", Math.max(0, completionTokens));
            metadata.put("tokens.decode", Math.max(0, decodeSteps));
            double prefillSec = prefillNanos / 1_000_000_000.0;
            double decodeSec = decodeNanos / 1_000_000_000.0;
            if (promptTokens > 0 && prefillSec > 1e-12) {
                metadata.put("bench.prefill_tps", promptTokens / prefillSec);
                metadata.put("bench.prefill_ms_per_token", roundMillis(prefillNanos) / promptTokens);
            }
            if (completionTokens > 0 && decodeSec > 1e-12) {
                metadata.put("bench.generation_tps", completionTokens / decodeSec);
            }
            if (decodeSteps > 0 && decodeNanos > 0L) {
                double decodeMsPerToken = roundMillis(decodeNanos) / decodeSteps;
                metadata.put("bench.decode_tps", decodeSteps / decodeSec);
                metadata.put("bench.decode_ms_per_token", decodeMsPerToken);
                metadata.put("bench.tpot_ms", decodeMsPerToken);
            }
        }
        return metadata;
    }

    String summary(String backend) {
        return String.format(Locale.ROOT,
                "backend=%s mode=%s load=%.2fms tokenize=%.2fms session=%.2fms ttft=%.2fms engine_ttft=%.2fms prefill=%.2fms decode=%.2fms tpot=%.2fms sampling=%.2fms attention=%.2fms ffn=%.2fms logits=%.2fms logits_copy=%.2fms steps=%d%s%s%s%s%s",
                backend, mode,
                roundMillis(modelLoadNanos),
                roundMillis(tokenizeNanos),
                roundMillis(sessionAllocateNanos),
                roundMillis(firstTokenNanos),
                roundMillis(engineFirstTokenNanos()),
                roundMillis(prefillNanos),
                roundMillis(decodeNanos),
                decodeSteps > 0 ? roundMillis(decodeNanos) / decodeSteps : 0.0,
                roundMillis(samplingNanos),
                roundMillis(attentionNanos),
                roundMillis(ffnNanos),
                roundMillis(logitsProjectionNanos),
                roundMillis(logitsMaterializationNanos),
                decodeSteps,
                hostLoadSummarySuffix(),
                linearSummarySuffix(),
                linearPathSummarySuffix(),
                ffnPathSummarySuffix(),
                attentionPathSummarySuffix());
    }

    private static double roundMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private long engineFirstTokenNanos() {
        if (firstTokenNanos <= 0L) {
            return 0L;
        }
        return Math.max(0L, firstTokenNanos - engineFirstTokenExcludedNanos());
    }

    private long engineFirstTokenExcludedNanos() {
        return Math.max(0L, modelLoadNanos)
                + Math.max(0L, tokenizeNanos)
                + Math.max(0L, sessionAllocateNanos);
    }

    private String linearSummarySuffix() {
        if (linearNanosByOperation.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" linear={");
        linearNanosByOperation.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder()))
                .limit(8)
                .forEachOrdered(entry -> {
                    if (sb.length() > " linear={".length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey())
                            .append('=')
                            .append(String.format(Locale.ROOT, "%.2f", roundMillis(entry.getValue())));
                });
        sb.append('}');
        return sb.toString();
    }

    private String linearPathSummarySuffix() {
        if (linearPathCounts.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" linear_paths={");
        linearPathCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue(Comparator.reverseOrder()))
                .limit(16)
                .forEachOrdered(entry -> {
                    if (sb.length() > " linear_paths={".length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey()).append('=').append(entry.getValue());
                });
        sb.append('}');
        return sb.toString();
    }

    private String ffnPathSummarySuffix() {
        if (ffnPathCounts.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" ffn_paths={");
        ffnPathCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue(Comparator.reverseOrder()))
                .limit(16)
                .forEachOrdered(entry -> {
                    if (sb.length() > " ffn_paths={".length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey()).append('=').append(entry.getValue());
                });
        sb.append('}');
        return sb.toString();
    }

    private HostLoadSnapshot hostLoadEnd() {
        if (hostLoadEnd == null) {
            hostLoadEnd = HostLoadSnapshot.capture();
        }
        return hostLoadEnd;
    }

    private String hostLoadSummarySuffix() {
        HostLoadSnapshot end = hostLoadEnd();
        if (!hostLoadStart.isAvailable() && !end.isAvailable()) {
            return "";
        }
        return String.format(Locale.ROOT,
                " host={load=%.2f/%.2f,cpu=%.0f%%,proc=%.0f%%,pressure=%s}",
                hostLoadStart.normalizedLoad(),
                end.normalizedLoad(),
                end.percentOrZero(end.systemCpuLoad()),
                end.percentOrZero(end.processCpuLoad()),
                end.pressureLabel());
    }

    private String attentionPathSummarySuffix() {
        if (attentionPathCounts.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" attention_paths={");
        attentionPathCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue(Comparator.reverseOrder()))
                .limit(8)
                .forEachOrdered(entry -> {
                    if (sb.length() > " attention_paths={".length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey()).append('=').append(entry.getValue());
                });
        sb.append('}');
        return sb.toString();
    }

    private static void putHostLoadMetadata(Map<String, Object> metadata,
                                            HostLoadSnapshot start,
                                            HostLoadSnapshot end) {
        if (metadata == null || start == null || end == null) {
            return;
        }
        metadata.put("host.available_processors", end.availableProcessors());
        metadata.put("host.load_average_start", start.systemLoadAverage());
        metadata.put("host.load_average_end", end.systemLoadAverage());
        metadata.put("host.load_average_per_cpu_start", start.normalizedLoad());
        metadata.put("host.load_average_per_cpu_end", end.normalizedLoad());
        metadata.put("host.cpu_load_percent", end.percentOrZero(end.systemCpuLoad()));
        metadata.put("host.process_cpu_load_percent", end.percentOrZero(end.processCpuLoad()));
        if (end.freeMemoryBytes() > 0L) {
            metadata.put("host.memory_free_mb", end.freeMemoryBytes() / (1024.0 * 1024.0));
        }
        if (end.totalMemoryBytes() > 0L) {
            metadata.put("host.memory_total_mb", end.totalMemoryBytes() / (1024.0 * 1024.0));
        }
        if (!Double.isNaN(end.freeMemoryPercent())) {
            metadata.put("host.memory_free_percent", end.freeMemoryPercent() * 100.0);
        }
        metadata.put("host.pressure", end.pressureLabel());
        metadata.put("host.pressure_warning", end.hasPressure());
    }

    private static String sanitizeMetricKey(String raw) {
        StringBuilder out = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char ch = raw.charAt(i);
            if ((ch >= 'a' && ch <= 'z')
                    || (ch >= 'A' && ch <= 'Z')
                    || (ch >= '0' && ch <= '9')) {
                out.append(Character.toLowerCase(ch));
            } else {
                out.append('_');
            }
        }
        return out.toString();
    }
}
