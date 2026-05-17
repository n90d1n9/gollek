package tech.kayys.gollek.ml.runner;

import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.Map;

/**
 * Result from a model inference call.
 *
 * @since 0.3.0
 */
public final class InferenceResult {

    private final Map<String, OutputTensor> outputs;
    private final Duration latency;
    private final long inputTokens;
    private final long outputTokens;

    private InferenceResult(Builder builder) {
        this.outputs = Map.copyOf(builder.outputs);
        this.latency = builder.latency;
        this.inputTokens = builder.inputTokens;
        this.outputTokens = builder.outputTokens;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Gets all output tensors by name.
     */
    public Map<String, OutputTensor> outputs() {
        return outputs;
    }

    /**
     * Gets the primary output tensor (first or named "output").
     */
    public OutputTensor primaryOutput() {
        if (outputs.containsKey("output")) return outputs.get("output");
        return outputs.values().iterator().next();
    }

    /**
     * Gets output by name.
     */
    public OutputTensor output(String name) {
        return outputs.get(name);
    }

    public Duration latency() { return latency; }
    public long inputTokens() { return inputTokens; }
    public long outputTokens() { return outputTokens; }

    /**
     * Output tensor from inference.
     */
    public record OutputTensor(
        String name,
        long[] shape,
        InferenceInput.InputDataType dtype,
        float[] floatData,
        int[] intData,
        MemorySegment nativeData
    ) {
        public float[] asFloats() {
            if (floatData != null) return floatData.clone();
            if (nativeData != null) {
                // Convert from native memory
                long numel = 1;
                for (long d : shape) numel *= d;
                float[] data = new float[(int) numel];
                for (int i = 0; i < numel; i++) {
                    data[i] = nativeData.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
                }
                return data;
            }
            return new float[0];
        }

        public long numel() {
            long n = 1;
            for (long d : shape) n *= d;
            return n;
        }
    }

    /**
     * Builder for InferenceResult.
     */
    public static class Builder {
        private Map<String, OutputTensor> outputs = Map.of();
        private Duration latency = Duration.ZERO;
        private long inputTokens = 0;
        private long outputTokens = 0;

        Builder() {}

        public Builder output(String name, OutputTensor tensor) {
            this.outputs = new java.util.HashMap<>(this.outputs);
            this.outputs.put(name, tensor);
            return this;
        }

        public Builder outputs(Map<String, OutputTensor> outputs) {
            this.outputs = new java.util.HashMap<>(outputs);
            return this;
        }

        public Builder latency(Duration latency) {
            this.latency = latency;
            return this;
        }

        public Builder latencyMs(long latencyMs) {
            this.latency = Duration.ofMillis(latencyMs);
            return this;
        }

        public Builder inputTokens(long inputTokens) {
            this.inputTokens = inputTokens;
            return this;
        }

        public Builder outputTokens(long outputTokens) {
            this.outputTokens = outputTokens;
            return this;
        }

        public InferenceResult build() {
            return new InferenceResult(this);
        }
    }
}
