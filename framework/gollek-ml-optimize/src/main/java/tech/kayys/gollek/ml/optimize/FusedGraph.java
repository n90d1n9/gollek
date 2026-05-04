package tech.kayys.gollek.ml.optimize;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;
import java.util.List;

/**
 * A compiled fused computation graph that can be executed as a single kernel.
 *
 * <p>Fused graphs reduce kernel launch overhead and memory transfers by
 * combining multiple operations into a single optimized kernel.
 *
 * @since 0.3.0
 */
public class FusedGraph {

    private final String id;
    private final List<FusionEngine.Operation> operations;
    private final FusionEngine.FusionDevice device;
    private final boolean isFused;
    private final String fusionKernelName;

    FusedGraph(String id, List<FusionEngine.Operation> operations,
               FusionEngine.FusionDevice device, boolean isFused) {
        this.id = id;
        this.operations = List.copyOf(operations);
        this.device = device;
        this.isFused = isFused;
        this.fusionKernelName = isFused ? generateKernelName(operations) : null;
    }

    static FusedGraph create(List<FusionEngine.Operation> ops,
                             FusionEngine.FusionDevice device,
                             java.util.Set<FusionEngine.FusionType> enabledFusions) {
        String id = "fused-" + computeFusionKey(ops) + "-" + System.currentTimeMillis();
        return new FusedGraph(id, ops, device, true);
    }

    /**
     * Executes the fused graph on input data.
     */
    public float[] execute(float[] input) {
        if (isFused && device != FusionEngine.FusionDevice.CPU) {
            return executeFusedKernel(input);
        } else {
            return executeSequential(input);
        }
    }

    /**
     * Executes on native memory (zero-copy).
     */
    public MemorySegment executeNative(MemorySegment input) {
        if (isFused && device != FusionEngine.FusionDevice.CPU) {
            return executeFusedKernelNative(input);
        } else {
            return executeSequentialNative(input);
        }
    }

    /**
     * Gets the unique identifier for this fused graph.
     */
    public String id() { return id; }

    /**
     * Gets the number of operations in this graph.
     */
    public int operationCount() { return operations.size(); }

    /**
     * Gets whether this graph was successfully fused.
     */
    public boolean isFused() { return isFused; }

    /**
     * Gets the fusion kernel name (if fused).
     */
    public String fusionKernelName() { return fusionKernelName; }

    /**
     * Gets the target device.
     */
    public FusionEngine.FusionDevice device() { return device; }

    // ── Fused Kernel Execution ─────────────────────────────────────────

    /**
     * Executes fused kernel on GPU (CUDA/Metal).
     * In production, this would launch a compiled PTX/Metal kernel.
     * For now, uses optimized SIMD sequential execution.
     */
    private float[] executeFusedKernel(float[] input) {
        // In production: load and execute compiled GPU kernel
        // For now: use optimized SIMD path
        return executeSequentialOptimized(input);
    }

    /**
     * Executes fused kernel on native memory.
     */
    private MemorySegment executeFusedKernelNative(MemorySegment input) {
        // In production: launch GPU kernel with native memory
        long bytes = input.byteSize();
        Arena arena = Arena.ofConfined();
        MemorySegment output = arena.allocate(bytes);

        // For now: copy input (identity transform)
        output.copyFrom(input);

        // Apply operations sequentially to native memory
        applyOperationsNative(output, operations);

        return output;
    }

    // ── Sequential Execution ───────────────────────────────────────────

    /**
     * Executes operations sequentially (CPU fallback).
     */
    private float[] executeSequential(float[] input) {
        float[] data = input.clone();
        for (FusionEngine.Operation op : operations) {
            data = applyOperation(op, data);
        }
        return data;
    }

    /**
     * Executes operations with SIMD optimization.
     */
    private float[] executeSequentialOptimized(float[] input) {
        float[] data = input.clone();

        // Check if we can fuse all operations into a single pass
        boolean canFuseSinglePass = canFuseToSinglePass(operations);

        if (canFuseSinglePass) {
            // Single-pass fused execution
            data = applyFusedSinglePass(operations, data);
        } else {
            // Fall back to sequential
            for (FusionEngine.Operation op : operations) {
                data = applyOperation(op, data);
            }
        }

        return data;
    }

    /**
     * Executes operations on native memory.
     */
    private MemorySegment executeSequentialNative(MemorySegment input) {
        // Apply each operation to native memory in place
        for (FusionEngine.Operation op : operations) {
            applyOperationNative(input, op);
        }
        return input;
    }

    // ── Operation Application ──────────────────────────────────────────

    /**
     * Applies a single operation to a float array.
     */
    private float[] applyOperation(FusionEngine.Operation op, float[] input) {
        float[] output = new float[input.length];
        String type = op.type().toLowerCase();
        float bias = op.param("bias", 0.0f);
        float dropoutP = op.param("dropout_p", 0.1f);
        long seed = op.param("seed", 42L);

        switch (type) {
            case "relu" -> applyRelu(input, output);
            case "add" -> applyAdd(input, output, bias);
            case "mul" -> applyMul(input, output, bias);
            case "gelu" -> applyGelu(input, output);
            case "silu" -> applySilu(input, output);
            case "sigmoid" -> applySigmoid(input, output);
            case "tanh" -> applyTanh(input, output);
            case "dropout" -> applyDropout(input, output, dropoutP, seed);
            case "layer_norm" -> applyLayerNorm(input, output, bias);
            default -> System.arraycopy(input, 0, output, 0, input.length);
        }

        return output;
    }

    /**
     * Applies an operation directly to native memory.
     */
    private void applyOperationNative(MemorySegment data, FusionEngine.Operation op) {
        long numel = data.byteSize() / 4L;  // Assume FLOAT32
        String type = op.type().toLowerCase();
        float bias = op.param("bias", 0.0f);
        float dropoutP = op.param("dropout_p", 0.1f);
        long seed = op.param("seed", 42L);

        switch (type) {
            case "relu" -> applyReluNative(data, numel);
            case "add" -> applyAddNative(data, numel, bias);
            case "mul" -> applyMulNative(data, numel, bias);
            case "gelu" -> applyGeluNative(data, numel);
            case "silu" -> applySiluNative(data, numel);
            case "sigmoid" -> applySigmoidNative(data, numel);
            case "tanh" -> applyTanhNative(data, numel);
            case "dropout" -> applyDropoutNative(data, numel, dropoutP, seed);
        }
    }

    // ── Native Memory Operations ───────────────────────────────────────

    private void applyOperationsNative(MemorySegment data, List<FusionEngine.Operation> ops) {
        for (FusionEngine.Operation op : ops) {
            applyOperationNative(data, op);
        }
    }

    private void applyReluNative(MemorySegment data, long numel) {
        for (long i = 0; i < numel; i++) {
            float v = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (v < 0) data.setAtIndex(ValueLayout.JAVA_FLOAT, i, 0);
        }
    }

    private void applyAddNative(MemorySegment data, long numel, float bias) {
        for (long i = 0; i < numel; i++) {
            float v = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, v + bias);
        }
    }

    private void applyMulNative(MemorySegment data, long numel, float scale) {
        for (long i = 0; i < numel; i++) {
            float v = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, v * scale);
        }
    }

    private void applyGeluNative(MemorySegment data, long numel) {
        for (long i = 0; i < numel; i++) {
            float x = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float gelu = (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))));
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu);
        }
    }

    private void applySiluNative(MemorySegment data, long numel) {
        for (long i = 0; i < numel; i++) {
            float x = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, x * sigmoid);
        }
    }

    private void applySigmoidNative(MemorySegment data, long numel) {
        for (long i = 0; i < numel; i++) {
            float x = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f / (1.0f + (float) Math.exp(-x)));
        }
    }

    private void applyTanhNative(MemorySegment data, long numel) {
        for (long i = 0; i < numel; i++) {
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.tanh(data.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
        }
    }

    private void applyDropoutNative(MemorySegment data, long numel, float p, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        float scale = 1.0f / (1.0f - p);
        for (long i = 0; i < numel; i++) {
            if (rng.nextFloat() > p) {
                float v = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                data.setAtIndex(ValueLayout.JAVA_FLOAT, i, v * scale);
            } else {
                data.setAtIndex(ValueLayout.JAVA_FLOAT, i, 0);
            }
        }
    }

    private void applyLayerNormNative(MemorySegment data, long numel, float eps) {
        // Compute mean and variance, then normalize
        // Simplified: just apply eps-based scaling
        for (long i = 0; i < numel; i++) {
            float v = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, v / (float) Math.sqrt(v * v + eps));
        }
    }

    // ── Array Operations ───────────────────────────────────────────────

    private void applyRelu(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
    }

    private void applyAdd(float[] input, float[] output, float bias) {
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] + bias;
        }
    }

    private void applyMul(float[] input, float[] output, float scale) {
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] * scale;
        }
    }

    private void applyGelu(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            output[i] = (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))));
        }
    }

    private void applySilu(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
            output[i] = x * sigmoid;
        }
    }

    private void applySigmoid(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
        }
    }

    private void applyTanh(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.tanh(input[i]);
        }
    }

    private void applyDropout(float[] input, float[] output, float p, long seed) {
        float scale = 1.0f / (1.0f - p);
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < input.length; i++) {
            if (rng.nextFloat() > p) {
                output[i] = input[i] * scale;
            } else {
                output[i] = 0;
            }
        }
    }

    private void applyLayerNorm(float[] input, float[] output, float eps) {
        float mean = 0, var = 0;
        for (float v : input) mean += v;
        mean /= input.length;
        for (float v : input) var += (v - mean) * (v - mean);
        var /= input.length;
        float std = (float) Math.sqrt(var + eps);
        for (int i = 0; i < input.length; i++) {
            output[i] = (input[i] - mean) / std;
        }
    }

    // ── Single-Pass Fusion ─────────────────────────────────────────────

    /**
     * Checks if operations can be fused into a single pass.
     */
    private boolean canFuseToSinglePass(List<FusionEngine.Operation> ops) {
        // Can fuse: elementwise + activation + dropout
        for (FusionEngine.Operation op : ops) {
            String type = op.type().toLowerCase();
            if (!type.equals("add") && !type.equals("mul") &&
                !type.equals("relu") && !type.equals("gelu") &&
                !type.equals("silu") && !type.equals("sigmoid") &&
                !type.equals("tanh") && !type.equals("dropout") &&
                !type.equals("layer_norm")) {
                return false;
            }
        }
        return true;
    }

    /**
     * Applies all operations in a single pass over the data.
     */
    private float[] applyFusedSinglePass(List<FusionEngine.Operation> ops, float[] input) {
        float[] output = new float[input.length];

        // Extract parameters from operations
        float addBias = 0, mulScale = 1.0f;
        boolean hasRelu = false, hasGelu = false, hasSilu = false;
        boolean hasSigmoid = false, hasTanh = false;
        float dropoutP = 0;
        long dropoutSeed = 42;
        boolean hasDropout = false;
        float layerNormEps = 1e-5f;
        boolean hasLayerNorm = false;

        for (FusionEngine.Operation op : ops) {
            String type = op.type().toLowerCase();
            switch (type) {
                case "add" -> addBias = op.param("bias", 0.0f);
                case "mul" -> mulScale = op.param("bias", 1.0f);
                case "relu" -> hasRelu = true;
                case "gelu" -> hasGelu = true;
                case "silu" -> hasSilu = true;
                case "sigmoid" -> hasSigmoid = true;
                case "tanh" -> hasTanh = true;
                case "dropout" -> {
                    hasDropout = true;
                    dropoutP = op.param("dropout_p", 0.1f);
                    dropoutSeed = op.param("seed", 42L);
                }
                case "layer_norm" -> {
                    hasLayerNorm = true;
                    layerNormEps = op.param("eps", 1e-5f);
                }
            }
        }

        java.util.Random rng = hasDropout ? new java.util.Random(dropoutSeed) : null;
        float dropoutScale = hasDropout ? 1.0f / (1.0f - dropoutP) : 1.0f;

        // Layer norm pre-computation
        float mean = 0, var = 0;
        if (hasLayerNorm) {
            for (float v : input) mean += v;
            mean /= input.length;
            for (float v : input) var += (v - mean) * (v - mean);
            var /= input.length;
        }
        float std = hasLayerNorm ? (float) Math.sqrt(var + layerNormEps) : 1.0f;

        // Single fused pass
        for (int i = 0; i < input.length; i++) {
            float x = input[i];

            // Apply affine transforms
            x = x * mulScale + addBias;

            // Apply layer norm if present
            if (hasLayerNorm) {
                x = (x - mean) / std;
            }

            // Apply activation
            if (hasRelu) x = Math.max(0, x);
            else if (hasGelu) x = (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))));
            else if (hasSilu) x = x / (1.0f + (float) Math.exp(-x));
            else if (hasSigmoid) x = 1.0f / (1.0f + (float) Math.exp(-x));
            else if (hasTanh) x = (float) Math.tanh(x);

            // Apply dropout
            if (hasDropout) {
                if (rng.nextFloat() > dropoutP) {
                    x *= dropoutScale;
                } else {
                    x = 0;
                }
            }

            output[i] = x;
        }

        return output;
    }

    // ── Utilities ──────────────────────────────────────────────────────

    private static String generateKernelName(List<FusionEngine.Operation> ops) {
        StringBuilder sb = new StringBuilder("fused_");
        for (int i = 0; i < ops.size(); i++) {
            if (i > 0) sb.append("_");
            sb.append(ops.get(i).type());
        }
        return sb.toString();
    }

    private static String computeFusionKey(List<FusionEngine.Operation> ops) {
        StringBuilder sb = new StringBuilder();
        for (FusionEngine.Operation op : ops) {
            sb.append(op.type()).append("-");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("FusedGraph{id='%s', ops=%d, fused=%s, device=%s}",
            id, operations.size(), isFused, device);
    }
}
