///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
// Enhanced Gollek LiteRT Edge Example - Demonstrates edge inference capabilities
//
// DEPENDENCY RESOLUTION:
//   cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-provider-core:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-error-code:0.1.0-SNAPSHOT
//JAVA_OPTIONS -Dquarkus.jbang.post-build=false --enable-native-access=ALL-UNNAMED
//JAVA 21+
//
// Usage:
//   jbang LiteRTEdgeExample.java --model /path/to/model.tflite
//   jbang LiteRTEdgeExample.java --delegate GPU --threads 4

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.LiteRTMetrics;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.sdk.litert.model.LiteRTModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * Gollek LiteRT Edge Inference Example
 * 
 * Demonstrates how to use LiteRT for edge device inference with:
 * - Model loading and management
 * - Synchronous and asynchronous inference
 * - Performance metrics
 * - Hardware delegate selection
 */
public class LiteRTEdgeExample {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       Gollek LiteRT Edge Inference Example              ║");
        ╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        String modelPath = System.getProperty("model", null);
        String delegateStr = System.getProperty("delegate", "AUTO");
        int numThreads = Integer.parseInt(System.getProperty("threads", "4"));
        boolean enableXnnpack = Boolean.parseBoolean(System.getProperty("xnnpack", "true"));

        // Parse CLI arguments if present (overrides system properties)
        for (int i = 0; i < args.length; i++) {
            if (("--model".equals(args[i]) || "-m".equals(args[i])) && i + 1 < args.length) {
                modelPath = args[++i];
            } else if (("--delegate".equals(args[i]) || "-d".equals(args[i])) && i + 1 < args.length) {
                delegateStr = args[++i];
            } else if (("--threads".equals(args[i]) || "-t".equals(args[i])) && i + 1 < args.length) {
                numThreads = Integer.parseInt(args[++i]);
            }
        }

        LiteRTConfig.Delegate delegate = LiteRTConfig.Delegate.valueOf(delegateStr.toUpperCase());

        LiteRTConfig config = LiteRTConfig.builder()
                .numThreads(numThreads)
                .delegate(delegate)
                .enableXnnpack(enableXnnpack)
                .useMemoryPool(true)
                .build();

        System.out.println("Configuration:");
        System.out.println("  Model:       " + (modelPath != null ? modelPath : "DEMO MODE"));
        System.out.println("  Delegate:    " + delegate);
        System.out.println("  Threads:     " + numThreads);
        System.out.println("  XNNPACK:     " + enableXnnpack);
        System.out.println();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            System.out.println("✓ LiteRT SDK initialized (Ver: " + sdk.getVersion() + ")");

            if (modelPath != null && !modelPath.isEmpty()) {
                if (modelPath.endsWith(".litertlm")) {
                    System.out.println("⚠️  Note: This model is a .litertlm language model container.");
                    System.out.println("   Generating text from Gemma-4 requires the LiteRT-LM engine.");
                    System.out.println();
                }
                runRealInference(sdk, modelPath);
            } else {
                runDemoInference(sdk);
            }

            displayMetrics(sdk);
        }

        System.out.println();
        System.out.println("✓ Example completed successfully");
    }

    private static void runRealInference(LiteRTSdk sdk, String modelPath) throws Exception {
        System.out.println("Loading model: " + modelPath);

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            System.err.println("❌ Error: Model file not found: " + modelPath);
            System.exit(1);
        }

        String modelId = "edge-model";
        try {
            sdk.loadModel(modelId, Path.of(modelPath));
            System.out.println("✓ Model loaded");

            LiteRTModelInfo modelInfo = sdk.getModelInfo(modelId);
            System.out.println("Model Info: " + modelInfo.getInputs().size() + " inputs, " + modelInfo.getOutputs().size() + " outputs");
            System.out.println();

            System.out.println("Running inference...");
            InferenceRequest request = InferenceRequest.builder()
                    .model(modelId)
                    .message(Message.user("Test inference"))
                    .build();

            long startTime = System.currentTimeMillis();
            InferenceResponse response = sdk.infer(request);
            long elapsed = System.currentTimeMillis() - startTime;

            System.out.println("✓ Inference successful in " + elapsed + "ms");
            System.out.println("  Content: " + response.getContent());
            System.out.println();
        } catch (Exception e) {
            System.err.println("❌ Inference Error: " + e.getMessage());
            if (e.getMessage() != null && e.getMessage().contains(".litertlm")) {
                System.err.println("   Please use LiteRT-LM enabled modules for specialized language inference.");
            }
        }
    }

    private static void runDemoInference(LiteRTSdk sdk) throws Exception {
        System.out.println("Running demo mode (synthetic data)");
        System.out.println("Tip: Use --model to run with a real .tflite or .litert model");
        System.out.println();

        for (int i = 0; i < 5; i++) {
            long startTime = System.currentTimeMillis();
            Thread.sleep(10 + (long) (Math.random() * 20));
            System.out.println("  Simulated step " + (i + 1) + " finished in " + (System.currentTimeMillis() - startTime) + "ms");
        }
    }

    private static void displayMetrics(LiteRTSdk sdk) {
        System.out.println("\nPerformance Summary:");
        try {
            LiteRTMetrics metrics = sdk.getMetrics(null);
            System.out.println("  Avg Latency: " + String.format("%.2f", metrics.getAvgLatencyMs()) + " ms");
            System.out.println("  Current Mem: " + formatBytes(metrics.getCurrentMemoryBytes()));
        } catch (Exception e) {
            System.out.println("  (Metrics available for real models only)");
        }
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
    }
}
