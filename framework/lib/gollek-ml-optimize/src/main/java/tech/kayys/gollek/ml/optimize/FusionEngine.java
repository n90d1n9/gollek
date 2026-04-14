package tech.kayys.gollek.ml.optimize;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Graph-level operation fusion engine that combines multiple operations
 * into single optimized kernels to reduce launch overhead and memory transfers.
 *
 * <h2>Problem</h2>
 * <p>Naive execution launches each operation separately:
 * <pre>
 *   matmul(W, x) → add(bias) → relu() → dropout(0.1)
 *   = 4 kernel launches + 3 memory transfers
 * </pre>
 *
 * <h2>Solution</h2>
 * <p>Fusion combines them into a single kernel:
 * <pre>
 *   fused_matmul_add_relu_dropout(W, x, bias, 0.1)
 *   = 1 kernel launch + 0 memory transfers
 * </pre>
 *
 * @since 0.3.0
 */
public class FusionEngine {

    private final FusionDevice device;
    private final Set<FusionType> enabledFusions;
    private final int minFusionSize;
    private final Map<String, Operation> operations = new ConcurrentHashMap<>();
    private final Map<String, FusedGraph> compiledGraphs = new ConcurrentHashMap<>();

    private FusionEngine(Builder builder) {
        this.device = builder.device;
        this.enabledFusions = Set.copyOf(builder.enabledFusions);
        this.minFusionSize = builder.minFusionSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Registers an operation for potential fusion.
     */
    public Operation registerOp(String type, Operation... inputs) {
        return registerOp(type, new long[0], inputs);
    }

    /**
     * Registers an operation with output size information.
     */
    public Operation registerOp(String type, long[] outputShape, Operation... inputs) {
        String opId = type + "-" + UUID.randomUUID().toString().substring(0, 8);
        Operation op = new Operation(opId, type, outputShape, List.of(inputs));
        operations.put(opId, op);
        return op;
    }

    /**
     * Creates a fusion group from multiple operations.
     */
    public FusionBuilder fuse(Operation... operations) {
        return new FusionBuilder(this, List.of(operations));
    }

    /**
     * Compiles a fused graph into an optimized kernel.
     */
    FusedGraph compileFusedGraph(List<Operation> ops) {
        if (!shouldFuse(ops)) {
            return new FusedGraph("unfused-" + UUID.randomUUID(), ops, device, false);
        }

        String fusionKey = computeFusionKey(ops);

        if (compiledGraphs.containsKey(fusionKey)) {
            return compiledGraphs.get(fusionKey);
        }

        FusedGraph fused = FusedGraph.create(ops, device, enabledFusions);
        compiledGraphs.put(fusionKey, fused);

        return fused;
    }

    /**
     * Checks if operations should be fused.
     */
    boolean shouldFuse(List<Operation> ops) {
        if (ops.isEmpty()) return false;

        // Check minimum size
        long totalElements = ops.stream()
            .mapToLong(Operation::outputSize)
            .sum();
        if (totalElements < minFusionSize) return false;

        // Check if all fusion types are enabled
        for (Operation op : ops) {
            FusionType requiredType = getFusionType(op.type());
            if (requiredType != null && !enabledFusions.contains(requiredType)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Computes a unique key for a fusion pattern.
     */
    String computeFusionKey(List<Operation> ops) {
        StringBuilder sb = new StringBuilder();
        for (Operation op : ops) {
            sb.append(op.type()).append("-");
        }
        return sb.toString();
    }

    /**
     * Maps operation type to fusion type.
     */
    private FusionType getFusionType(String opType) {
        return switch (opType.toLowerCase()) {
            case "add", "sub", "mul", "div", "pow" -> FusionType.ELEMENTWISE;
            case "relu", "gelu", "silu", "sigmoid", "tanh" -> FusionType.ACTIVATION;
            case "batch_norm", "layer_norm", "rms_norm" -> FusionType.NORMALIZATION;
            case "matmul", "conv2d", "linear" -> FusionType.MATMUL;
            case "dropout" -> FusionType.REGULARIZATION;
            default -> null;
        };
    }

    /**
     * Supported fusion types.
     */
    public enum FusionType {
        ELEMENTWISE,
        ACTIVATION,
        NORMALIZATION,
        MATMUL,
        REGULARIZATION
    }

    /**
     * Target device for fusion.
     */
    public enum FusionDevice {
        CPU, CUDA, METAL, ROCM, AUTO
    }

    /**
     * Builder for FusionEngine.
     */
    public static class Builder {
        private FusionDevice device = FusionDevice.AUTO;
        private final Set<FusionType> enabledFusions = EnumSet.allOf(FusionType.class);
        private int minFusionSize = 1024;

        Builder() {}

        public Builder device(FusionDevice device) { this.device = device; return this; }
        public Builder enableFusion(FusionType type) { this.enabledFusions.add(type); return this; }
        public Builder disableFusion(FusionType type) { this.enabledFusions.remove(type); return this; }
        public Builder minFusionSize(int minFusionSize) { this.minFusionSize = minFusionSize; return this; }

        public FusionEngine build() {
            return new FusionEngine(this);
        }
    }

    /**
     * Represents a single operation in the computation graph.
     */
    public record Operation(
        String id,
        String type,
        long[] outputShape,
        List<Operation> inputs,
        Map<String, Object> params
    ) {
        public Operation(String id, String type, long[] outputShape, List<Operation> inputs) {
            this(id, type, outputShape, inputs, Map.of());
        }

        /**
         * Gets the number of elements in the output tensor.
         */
        public long outputSize() {
            if (outputShape.length > 0) {
                long n = 1;
                for (long d : outputShape) n *= d;
                return n;
            }
            // Fallback: estimate from inputs
            if (!inputs.isEmpty()) {
                return inputs.get(0).outputSize();
            }
            return 0;
        }

        /**
         * Gets a parameter value with type safety.
         */
        @SuppressWarnings("unchecked")
        public <T> T param(String key, T defaultValue) {
            return (T) params.getOrDefault(key, defaultValue);
        }

        /**
         * Creates an operation with parameters.
         */
        public Operation withParams(Map<String, Object> params) {
            return new Operation(id, type, outputShape, inputs, params);
        }
    }

    /**
     * Builder for creating fusion groups.
     */
    public static class FusionBuilder {
        private final FusionEngine engine;
        private final List<Operation> operations;
        private Map<String, Object> fusedParams = Map.of();

        FusionBuilder(FusionEngine engine, List<Operation> operations) {
            this.engine = engine;
            this.operations = operations;
        }

        /**
         * Adds parameters to the fused operation.
         */
        public FusionBuilder withParams(Map<String, Object> params) {
            this.fusedParams = new HashMap<>(this.fusedParams);
            this.fusedParams.putAll(params);
            return this;
        }

        /**
         * Adds a single parameter.
         */
        public FusionBuilder withParam(String key, Object value) {
            this.fusedParams = new HashMap<>(this.fusedParams);
            this.fusedParams.put(key, value);
            return this;
        }

        /**
         * Compiles the fusion group into an optimized kernel.
         */
        public FusedGraph compile() {
            // Apply fused params to operations
            List<Operation> ops = operations.stream()
                .map(op -> op.withParams(fusedParams))
                .toList();
            return engine.compileFusedGraph(ops);
        }
    }
}
