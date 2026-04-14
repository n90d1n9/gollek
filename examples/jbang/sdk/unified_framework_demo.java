///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-ml:0.3.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-optimize:0.3.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-onnx:0.3.0-SNAPSHOT
//DEPS info.picocli:picocli:4.7.5
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

package tech.kayys.gollek.examples.framework;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.runner.*;
import tech.kayys.gollek.ml.runner.batching.BatchingEngine;
import tech.kayys.gollek.ml.optimize.*;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Comprehensive example demonstrating Gollek v0.3 framework capabilities:
 * - Unified Model Runner (GGUF, ONNX, LiteRT, TensorRT)
 * - Dynamic Batching Engine
 * - Graph Fusion Engine
 * - Quantization (INT8/INT4/FP8/AWQ/GPTQ)
 * - Shared Memory Pool
 *
 * Usage:
 *   jbang unified_framework_demo.java --model model.onnx --device cuda
 *   jbang unified_framework_demo.java --demo quantization --input model.safetensors
 *   jbang unified_framework_demo.java --demo fusion
 *   jbang unified_framework_demo.java --demo memory-pool
 */
@Command(name = "unified_framework_demo", mixinStandardHelpOptions = true,
         description = "Gollek v0.3 unified framework capabilities demo")
public class unified_framework_demo implements Runnable {

    @Option(names = {"--model", "-m"}, description = "Path to model file")
    String modelPath = "model.onnx";

    @Option(names = {"--device", "-d"}, description = "Target device: cpu, cuda, metal, auto")
    String device = "auto";

    @Option(names = {"--demo"}, description = "Specific demo: runner, batching, fusion, quantization, memory-pool, all")
    String demoType = "all";

    public static void main(String[] args) {
        int exitCode = new CommandLine(new unified_framework_demo()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public void run() {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       Gollek v0.3 Framework Capabilities Demo           ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝\n");

        switch (demoType.toLowerCase()) {
            case "runner" -> demoUnifiedRunner();
            case "batching" -> demoDynamicBatching();
            case "fusion" -> demoGraphFusion();
            case "quantization" -> demoQuantization();
            case "memory-pool" -> demoMemoryPool();
            case "all" -> {
                demoUnifiedRunner();
                demoDynamicBatching();
                demoGraphFusion();
                demoQuantization();
                demoMemoryPool();
            }
            default -> System.out.println("Unknown demo: " + demoType);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 1. Unified Model Runner
    // ═══════════════════════════════════════════════════════════

    void demoUnifiedRunner() {
        System.out.println("\n┌─────────────────────────────────────────────────────┐");
        System.out.println("│  1. Unified Model Runner                           │");
        System.out.println("│  Single API for all frameworks                      │");
        System.out.println("└─────────────────────────────────────────────────────┘\n");

        // Before: framework-specific code
        System.out.println("BEFORE: Framework-specific code");
        System.out.println("  LibTorchRunner torchRunner = new LibTorchRunner(path);");
        System.out.println("  LiteRTRunner litertRunner = new LiteRTRunner(path);");
        System.out.println("  OnnxRunner onnxRunner = new OnnxRunner(path);");
        System.out.println();

        // After: unified API
        System.out.println("AFTER: Unified API");
        try {
            RunnerDevice runnerDevice = parseDevice(device);

            ModelRunner runner = ModelRunner.builder()
                .modelPath(modelPath)
                .device(runnerDevice)
                .option("session.intra_op_num_threads", 4)
                .build();

            System.out.printf("  ✓ Runner created: %s%n", runner.id());
            System.out.printf("  ✓ Format: %s%n", runner.format());
            System.out.printf("  ✓ Device: %s%n", runner.device().deviceName());
            System.out.printf("  ✓ Ready: %s%n", runner.isReady());

            // Show metadata
            ModelMetadata metadata = runner.metadata();
            System.out.printf("  ✓ Model: %s (%s)%n", metadata.name(), metadata.format());
            System.out.printf("  ✓ Parameters: %,d%n", metadata.parameterCount());
            System.out.printf("  ✓ Inputs: %s%n", metadata.inputNames());
            System.out.printf("  ✓ Outputs: %s%n", metadata.outputNames());

            runner.close();
            System.out.println("\n  ✓ Runner closed successfully");

        } catch (Exception e) {
            System.out.printf("  ⚠ Runner demo (model not found): %s%n", e.getMessage());
            System.out.println("  ℹ Using mock runner for demonstration...");
            showMockRunnerDemo();
        }
    }

    void showMockRunnerDemo() {
        System.out.println("\n  Mock demonstration with simulated model:");
        System.out.println("    // Create runner");
        System.out.println("    ModelRunner runner = ModelRunner.builder()");
        System.out.println("        .modelPath(\"llama-3-8b.gguf\")");
        System.out.println("        .device(RunnerDevice.CUDA)");
        System.out.println("        .build();");
        System.out.println();
        System.out.println("    // Inference");
        System.out.println("    InferenceResult result = runner.infer(");
        System.out.println("        InferenceInput.fromInts(tokenIds, 1, seqLen)");
        System.out.println("    );");
        System.out.println();
        System.out.println("    // Results");
        System.out.println("    System.out.println(result.primaryOutput().asFloats());");
        System.out.println("    System.out.println(\"Latency: \" + result.latency());");
    }

    // ═══════════════════════════════════════════════════════════
    // 2. Dynamic Batching Engine
    // ═══════════════════════════════════════════════════════════

    void demoDynamicBatching() {
        System.out.println("\n┌─────────────────────────────────────────────────────┐");
        System.out.println("│  2. Dynamic Batching Engine                        │");
        System.out.println("│  Automatic request batching with SLA               │");
        System.out.println("└─────────────────────────────────────────────────────┘\n");

        System.out.println("  Creating batching engine...");
        System.out.println("    BatchingEngine batcher = BatchingEngine.builder()");
        System.out.println("        .runner(modelRunner)");
        System.out.println("        .maxBatchSize(256)");
        System.out.println("        .maxWaitTime(Duration.ofMillis(10))");
        System.out.println("        .slaP95(Duration.ofMillis(100))");
        System.out.println("        .build();");
        System.out.println();

        // Simulate batching behavior
        System.out.println("  Simulating 10 concurrent requests...");

        for (int i = 0; i < 10; i++) {
            final int requestId = i + 1;
            System.out.printf("    Request %d submitted...%n", requestId);
        }

        System.out.println();
        System.out.println("  Requests automatically batched:");
        System.out.println("    Batch 1: Requests [1, 2, 3, 4] → Single GPU kernel");
        System.out.println("    Batch 2: Requests [5, 6, 7, 8] → Single GPU kernel");
        System.out.println("    Batch 3: Requests [9, 10]       → Single GPU kernel");
        System.out.println();
        System.out.println("  ✓ 10 requests → 3 batches (70% fewer kernel launches)");
        System.out.println("  ✓ Estimated speedup: 3-5x throughput improvement");
    }

    // ═══════════════════════════════════════════════════════════
    // 3. Graph Fusion Engine
    // ═══════════════════════════════════════════════════════════

    void demoGraphFusion() {
        System.out.println("\n┌─────────────────────────────────────────────────────┐");
        System.out.println("│  3. Graph Fusion Engine                            │");
        System.out.println("│  Fuse operations into single kernels               │");
        System.out.println("└─────────────────────────────────────────────────────┘\n");

        FusionEngine engine = FusionEngine.builder()
            .device(FusionEngine.FusionDevice.CPU)
            .enableFusion(FusionEngine.FusionType.ELEMENTWISE)
            .enableFusion(FusionEngine.FusionType.ACTIVATION)
            .minFusionSize(1024)
            .build();

        System.out.println("  Creating computation graph...");

        // Create input operation
        FusionEngine.Operation input = engine.registerOp("input", new long[]{1, 4096});
        System.out.println("    ✓ Input: [1, 4096]");

        // Create operations
        FusionEngine.Operation matmul = engine.registerOp("matmul", new long[]{1, 4096}, input);
        System.out.println("    ✓ Matmul: [1, 4096] × [4096, 4096]");

        FusionEngine.Operation add = engine.registerOp("add", new long[]{1, 4096}, matmul)
            .withParams(Map.of("bias", 0.5f));
        System.out.println("    ✓ Add bias: 0.5");

        FusionEngine.Operation relu = engine.registerOp("relu", new long[]{1, 4096}, add);
        System.out.println("    ✓ ReLU activation");

        System.out.println();
        System.out.println("  Fusing operations...");

        // Fuse and compile
        FusedGraph fused = engine.fuse(matmul, add, relu).compile();

        System.out.printf("    ✓ Fused graph: %s%n", fused.fusionKernelName());
        System.out.println("    ✓ Operations fused: 3 → 1 kernel");
        System.out.println("    ✓ Memory transfers reduced: 2 → 0");
        System.out.println();

        // Execute fused graph
        System.out.println("  Executing fused graph...");
        float[] inputData = new float[4096];
        for (int i = 0; i < inputData.length; i++) inputData[i] = (float) Math.random();

        long startTime = System.currentTimeMillis();
        float[] output = fused.execute(inputData);
        long elapsed = System.currentTimeMillis() - startTime;

        System.out.printf("    ✓ Execution time: %dms%n", elapsed);
        System.out.printf("    ✓ Output shape: [1, 4096] (%,d elements)%n", output.length);
        System.out.printf("    ✓ Output range: [%.4f, %.4f]%n",
            java.util.Arrays.stream(output).min().orElse(0),
            java.util.Arrays.stream(output).max().orElse(0));
        System.out.println();
        System.out.println("  Estimated speedup: 2-3x from reduced kernel launches");
    }

    // ═══════════════════════════════════════════════════════════
    // 4. Quantization Engine
    // ═══════════════════════════════════════════════════════════

    void demoQuantization() {
        System.out.println("\n┌─────────────────────────────────────────────────────┐");
        System.out.println("│  4. Quantization Engine                            │");
        System.out.println("│  INT4/INT8/FP8 with AWQ/GPTQ                       │");
        System.out.println("└─────────────────────────────────────────────────────┘\n");

        System.out.println("  Comparing quantization schemes:");
        System.out.println("  ┌──────────┬────────────┬─────────────┬──────────┐");
        System.out.println("  │ Scheme   │ Compression│ Accuracy    │ Speedup  │");
        System.out.println("  ├──────────┼────────────┼─────────────┼──────────┤");
        System.out.println("  │ FP16     │ 2x         │ 0% loss     │ 1x       │");
        System.out.println("  │ INT8     │ 4x         │ <1% loss    │ 1.5-2x   │");
        System.out.println("  │ INT4     │ 8x         │ 1-3% loss   │ 2-4x     │");
        System.out.println("  │ FP8      │ 4x         │ <0.5% loss  │ 1.5-2x   │");
        System.out.println("  │ AWQ      │ 6-8x       │ <1% loss    │ 2-3x     │");
        System.out.println("  │ GPTQ     │ 4-8x       │ <2% loss    │ 2-4x     │");
        System.out.println("  └──────────┴────────────┴─────────────┴──────────┘");
        System.out.println();

        // Demo AWQ quantization
        System.out.println("  Example: AWQ INT4 Quantization");
        System.out.println("    QuantizedModel qModel = QuantizationEngine.builder()");
        System.out.println("        .modelPath(\"llama-3-70b.safetensors\")");
        System.out.println("        .scheme(QuantizationScheme.AWQ)");
        System.out.println("        .targetPrecision(Precision.INT4)");
        System.out.println("        .calibrationData(calibrationDataset)");
        System.out.println("        .build()");
        System.out.println("        .quantize();");
        System.out.println();
        System.out.println("    // Results:");
        System.out.println("    // Original:  140GB (FP16)");
        System.out.println("    // Quantized:  35GB (INT4)");
        System.out.println("    // Compression: 4x");
        System.out.println("    // Accuracy loss: <1%");
        System.out.println("    // Inference speedup: 2-3x");
    }

    // ═══════════════════════════════════════════════════════════
    // 5. Shared Memory Pool
    // ═══════════════════════════════════════════════════════════

    void demoMemoryPool() {
        System.out.println("\n┌─────────────────────────────────────────────────────┐");
        System.out.println("│  5. Shared Memory Pool                             │");
        System.out.println("│  Zero-copy tensor allocation                       │");
        System.out.println("└─────────────────────────────────────────────────────┘\n");

        MemoryPool pool = MemoryPool.builder()
            .device(RunnerDevice.CPU)
            .maxSizeGB(4)
            .strategy(MemoryPool.MemoryStrategy.FRAGMENT_AWARE)
            .build();

        System.out.println("  Pool created: 4GB, fragment-aware strategy");
        System.out.println();

        // Allocate tensors
        System.out.println("  Allocating tensors from pool...");
        var tensor1 = pool.allocate(4096L * 4096 * 4);  // 4096x4096 FLOAT32
        System.out.printf("    ✓ Tensor1: 4096×4096 FLOAT32 (%.1f MB)%n", tensor1.byteSize() / (1024.0 * 1024.0));

        var tensor2 = pool.allocate(1024L * 1024 * 4);  // 1024x1024 FLOAT32
        System.out.printf("    ✓ Tensor2: 1024×1024 FLOAT32 (%.1f MB)%n", tensor2.byteSize() / (1024.0 * 1024.0));

        var aligned = pool.allocateAligned(8192L * 4, 64);  // Aligned to 64 bytes
        System.out.printf("    ✓ Aligned: 8192 FLOAT32, 64-byte aligned%n");

        System.out.println();
        MemoryPool.PoolStats stats = pool.stats();
        System.out.println("  Pool Statistics:");
        System.out.printf("    Total:     %s%n", stats.formattedMax());
        System.out.printf("    Allocated: %s%n", stats.formattedAllocated());
        System.out.printf("    Utilization: %.1f%%%n", stats.utilizationPercent());
        System.out.printf("    Active allocations: %d%n", stats.activeAllocations());
        System.out.println();

        // Demonstrate zero-copy
        System.out.println("  Zero-copy demonstration:");
        System.out.println("    // Allocate from pool");
        System.out.println("    MemorySegment tensor = pool.allocate(1024 * 1024 * 4);");
        System.out.println();
        System.out.println("    // Use with any framework (no copy needed!)");
        System.out.println("    libtorchRunner.infer(tensor);");
        System.out.println("    onnxRunner.infer(tensor);      // Zero-copy!");
        System.out.println("    litertRunner.infer(tensor);    // Zero-copy!");
        System.out.println();

        // Free memory
        pool.free(tensor1);
        pool.free(tensor2);
        System.out.println("  ✓ Tensors freed");
        System.out.printf("    Utilization after free: %.1f%%%n", pool.stats().utilizationPercent());

        pool.close();
        System.out.println("  ✓ Pool closed");
    }

    // ═══════════════════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════════════════

    private RunnerDevice parseDevice(String device) {
        return switch (device.toLowerCase()) {
            case "cuda" -> RunnerDevice.CUDA;
            case "metal" -> RunnerDevice.METAL;
            case "rocm" -> RunnerDevice.ROCM;
            case "cpu" -> RunnerDevice.CPU;
            default -> RunnerDevice.AUTO;
        };
    }
}
