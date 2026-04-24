package tech.kayys.gollek.inference.nativeimpl;

import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

/**
 * Tracks performance and utilization metrics for the native inference engine.
 */
public final class InferenceMetrics {
    private final LongAdder tokensProcessed = new LongAdder();
    private final LongAdder totalInferenceTimeNs = new LongAdder();
    private final LongAdder totalPromptTimeNs = new LongAdder();
    
    private final AtomicLongArray expertUtilization;
    private final int numExperts;

    public InferenceMetrics(int numExperts) {
        this.numExperts = numExperts;
        this.expertUtilization = numExperts > 0 ? new AtomicLongArray(numExperts) : null;
    }

    public void recordToken(long timeNs) {
        tokensProcessed.increment();
        totalInferenceTimeNs.add(timeNs);
    }

    public void recordPrompt(long timeNs) {
        totalPromptTimeNs.add(timeNs);
    }

    public void recordExpertActivation(int expertIdx) {
        if (expertUtilization != null && expertIdx < numExperts) {
            expertUtilization.incrementAndGet(expertIdx);
        }
    }

    public long getTokensProcessed() { return tokensProcessed.sum(); }
    public long getTotalInferenceTimeNs() { return totalInferenceTimeNs.sum(); }
    public long getTotalPromptTimeNs() { return totalPromptTimeNs.sum(); }

    public double getTokensPerSecond() {
        long tokens = getTokensProcessed();
        long time = getTotalInferenceTimeNs();
        if (time == 0) return 0;
        return (double) tokens / (time / 1_000_000_000.0);
    }

    public double getExpertUtilization(int expertIdx) {
        if (expertUtilization == null) return 0;
        long activations = expertUtilization.get(expertIdx);
        long total = tokensProcessed.sum();
        if (total == 0) return 0;
        return (double) activations / total;
    }

    public void reset() {
        tokensProcessed.reset();
        totalInferenceTimeNs.reset();
        totalPromptTimeNs.reset();
        if (expertUtilization != null) {
            for (int i = 0; i < numExperts; i++) expertUtilization.set(i, 0);
        }
    }
}
