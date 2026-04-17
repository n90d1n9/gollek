package tech.kayys.gollek.inference.nativeimpl;

import java.util.HashMap;
import java.util.Map;

/**
 * Performance metrics for a single inference request.
 * Parity with llama.cpp timings.
 */
public record InferenceMetrics(
    long loadTimeNanos,
    long prefillTimeNanos,
    long decodeTimeNanos,
    long firstTokenTimeNanos,
    int inputTokens,
    int outputTokens
) {
    public double getPrefillSpeed() {
        if (prefillTimeNanos == 0) return 0;
        return (inputTokens * 1_000_000_000.0) / prefillTimeNanos;
    }

    public double getGenerationSpeed() {
        if (decodeTimeNanos == 0) return 0;
        return (outputTokens * 1_000_000_000.0) / decodeTimeNanos;
    }

    public double getTtftMs() {
        return (firstTokenTimeNanos - (prefillTimeNanos > 0 ? (firstTokenTimeNanos - decodeTimeNanos) : 0)) / 1_000_000.0; 
        // Simple TTFT delta from start of prefill
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> meta = new HashMap<>();
        meta.put("bench.load_ms", loadTimeNanos / 1_000_000.0);
        meta.put("bench.prefill_ms", prefillTimeNanos / 1_000_000.0);
        meta.put("bench.decode_ms", decodeTimeNanos / 1_000_000.0);
        meta.put("bench.prefill_tps", getPrefillSpeed());
        meta.put("bench.generation_tps", getGenerationSpeed());
        meta.put("bench.ttft_ms", (firstTokenTimeTime() - startTime()) / 1_000_000.0); // I'll refine this in the provider
        return meta;
    }

    // Helper for easier calculation
    private long startTime() { return 0; } // Placeholder for logic
    private long firstTokenTimeTime() { return 0; }
}
