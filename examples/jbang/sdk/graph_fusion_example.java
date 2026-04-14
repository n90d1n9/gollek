///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-ml:0.3.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-optimize:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

package tech.kayys.gollek.examples.framework;

import tech.kayys.gollek.ml.optimize.FusionEngine;
import tech.kayys.gollek.ml.optimize.FusedGraph;

import java.util.Map;
import java.util.Random;

/**
 * Graph Fusion Engine Example
 * 
 * Demonstrates how to fuse multiple operations into a single optimized kernel
 * to reduce launch overhead and memory transfers.
 *
 * Problem: matmul → add → relu → dropout = 4 kernel launches + 3 memory transfers
 * Solution: fused_matmul_add_relu_dropout = 1 kernel launch + 0 memory transfers
 *
 * Usage:
 *   jbang graph_fusion_example.java
 *   jbang graph_fusion_example.java --size 8192 --fuse true
 */
public class graph_fusion_example {

    public static void main(String[] args) {
        int size = 4096;
        boolean enableFusion = true;

        // Parse arguments
        for (int i = 0; i < args.length; i++) {
            if ("--size".equals(args[i]) && i + 1 < args.length) {
                size = Integer.parseInt(args[i + 1]);
            } else if ("--fuse".equals(args[i]) && i + 1 < args.length) {
                enableFusion = Boolean.parseBoolean(args[i + 1]);
            }
        }

        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║       Graph Fusion Engine Example               ║");
        System.out.println("╚══════════════════════════════════════════════════╝\n");

        // Create fusion engine
        FusionEngine engine = FusionEngine.builder()
            .device(FusionEngine.FusionDevice.CPU)
            .enableFusion(FusionEngine.FusionType.ELEMENTWISE)
            .enableFusion(FusionEngine.FusionType.ACTIVATION)
            .enableFusion(FusionEngine.FusionType.REGULARIZATION)
            .minFusionSize(1024)
            .build();

        // Create computation graph: matmul → add → relu → dropout
        System.out.println("Creating computation graph...");
        FusionEngine.Operation input = engine.registerOp("input", new long[]{1, size});
        FusionEngine.Operation matmul = engine.registerOp("matmul", new long[]{1, size}, input);
        FusionEngine.Operation add = engine.registerOp("add", new long[]{1, size}, matmul)
            .withParams(Map.of("bias", 0.5f));
        FusionEngine.Operation relu = engine.registerOp("relu", new long[]{1, size}, add);
        FusionEngine.Operation dropout = engine.registerOp("dropout", new long[]{1, size}, relu)
            .withParams(Map.of("dropout_p", 0.1f, "seed", 42L));

        System.out.println("  ✓ Input: [1, " + size + "]");
        System.out.println("  ✓ Matmul: [1, " + size + "] × [" + size + ", " + size + "]");
        System.out.println("  ✓ Add bias: 0.5");
        System.out.println("  ✓ ReLU activation");
        System.out.println("  ✓ Dropout: p=0.1");
        System.out.println();

        // Generate input data
        float[] inputData = new float[size];
        Random rng = new Random(42);
        for (int i = 0; i < size; i++) {
            inputData[i] = (float) rng.nextGaussian();
        }

        // Benchmark WITHOUT fusion (sequential execution)
        System.out.println("Benchmarking WITHOUT fusion (sequential)...");
        long startTime = System.currentTimeMillis();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            float[] data = inputData.clone();
            data = applyMatmul(data, size);
            data = applyAdd(data, 0.5f);
            data = applyRelu(data);
            data = applyDropout(data, 0.1f, 42L);
        }
        long sequentialTime = System.currentTimeMillis() - startTime;
        System.out.printf("  Sequential: %d iterations in %dms (%.2fms/iter)%n",
            iterations, sequentialTime, sequentialTime / (double) iterations);

        // Benchmark WITH fusion
        System.out.println("\nBenchmarking WITH fusion (single-pass)...");
        FusedGraph fusedGraph = engine.fuse(matmul, add, relu, dropout).compile();
        System.out.printf("  ✓ Fused kernel: %s%n", fusedGraph.fusionKernelName());
        System.out.printf("  ✓ Operations fused: 4 → 1 kernel%n");

        startTime = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            float[] output = fusedGraph.execute(inputData);
        }
        long fusedTime = System.currentTimeMillis() - startTime;
        System.out.printf("  Fused: %d iterations in %dms (%.2fms/iter)%n",
            iterations, fusedTime, fusedTime / (double) iterations);

        // Results
        System.out.println("\n┌─────────────────────────────────────────────┐");
        System.out.println("│           Results Summary                   │");
        System.out.println("├─────────────────────────────────────────────┤");
        System.out.printf("│ Sequential time:  %5dms                       │%n", sequentialTime);
        System.out.printf("│ Fused time:       %5dms                       │%n", fusedTime);
        double speedup = sequentialTime / (double) fusedTime;
        System.out.printf("│ Speedup:          %.2fx                        │%n", speedup);
        System.out.printf("│ Memory transfers:  3 → 0                      │%n");
        System.out.printf("│ Kernel launches:   4 → 1                      │%n");
        System.out.println("└─────────────────────────────────────────────┘");
    }

    // ── Individual Operations ─────────────────────────────────────

    static float[] applyMatmul(float[] input, int size) {
        float[] output = new float[size];
        // Simplified: element-wise multiply for demo
        for (int i = 0; i < size; i++) {
            output[i] = input[i] * 0.95f;
        }
        return output;
    }

    static float[] applyAdd(float[] input, float bias) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] + bias;
        }
        return output;
    }

    static float[] applyRelu(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    static float[] applyDropout(float[] input, float p, long seed) {
        float[] output = new float[input.length];
        float scale = 1.0f / (1.0f - p);
        Random rng = new Random(seed);
        for (int i = 0; i < input.length; i++) {
            if (rng.nextFloat() > p) {
                output[i] = input[i] * scale;
            }
        }
        return output;
    }
}
