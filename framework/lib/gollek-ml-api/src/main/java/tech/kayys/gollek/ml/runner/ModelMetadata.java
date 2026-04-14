package tech.kayys.gollek.ml.runner;

import java.util.List;
import java.util.Map;

/**
 * Metadata about a loaded model.
 *
 * @since 0.3.0
 */
public record ModelMetadata(
    /** Model name/identifier */
    String name,

    /** Model format (GGUF, ONNX, etc.) */
    String format,

    /** Input tensor specifications */
    Map<String, InputSpec> inputs,

    /** Output tensor specifications */
    Map<String, OutputSpec> outputs,

    /** Model parameters count */
    long parameterCount,

    /** Framework-specific metadata */
    Map<String, Object> extra
) {
    public record InputSpec(
        String name,
        long[] shape,
        String dtype,
        boolean dynamicBatch
    ) {}

    public record OutputSpec(
        String name,
        long[] shape,
        String dtype
    ) {}

    /**
     * Gets the expected batch size from input shape.
     */
    public int expectedBatchSize() {
        InputSpec firstInput = inputs.values().iterator().next();
        return firstInput.shape.length > 0 ? (int) firstInput.shape[0] : 1;
    }

    /**
     * Gets the expected sequence length from input shape.
     */
    public int expectedSequenceLength() {
        InputSpec firstInput = inputs.values().iterator().next();
        return firstInput.shape.length > 1 ? (int) firstInput.shape[1] : 0;
    }

    /**
     * Gets input names in order.
     */
    public List<String> inputNames() {
        return inputs.keySet().stream().toList();
    }

    /**
     * Gets output names in order.
     */
    public List<String> outputNames() {
        return outputs.keySet().stream().toList();
    }

    /**
     * Empty metadata for testing/stubbing.
     */
    public static ModelMetadata empty() {
        return new ModelMetadata(
            "unknown", "unknown",
            Map.of(), Map.of(),
            0, Map.of()
        );
    }
}
