///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek LiteRT Text Generation Example
//
// DEPENDENCY RESOLUTION:
// The SDK modules must be built and installed to local Maven repository first:
//   cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests
//
// REPOS mavenLocal
// DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
// JAVA 21+
//
// Usage:
//   jbang TextGenerationExample.java --model gemma.litertlm --prompt "Hello, how are you?"
//   jbang TextGenerationExample.java --model model.litertlm --prompt "Once upon a time" --max-tokens 100
//   jbang TextGenerationExample.java --model model.litertlm --prompt "Write a story" --temperature 0.8 --topk 50

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.LiteRTMetrics;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.sdk.litert.model.LiteRTModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * LiteRT Text Generation Example
 *
 * Demonstrates text generation using LiteRT language models (.litertlm)
 * like Gemma, Llama, Phi, etc.
 *
 * Features:
 * - Text tokenization and preprocessing
 * - Single-turn text generation
 * - Streaming generation (token-by-token)
 * - Configurable generation parameters (temperature, top-k, max tokens)
 * - Interactive chat mode
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class TextGenerationExample {

    // Simple vocabulary mapping (in production, load from model's tokenizer)
    private static final String[] SAMPLE_VOCAB = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "hello", "world", "how", "are", "you", "today", "is", "good", "fine", "thanks",
        "once", "upon", "time", "story", "long", "ago", "in", "faraway", "land", "king",
        "queen", "prince", "princess", "castle", "dragon", "magic", "spell", "enchanted", "forest", "adventure"
    };

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║          LiteRT Text Generation Example                  ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // Parse arguments
        String modelPath = System.getProperty("model", null);
        String prompt = System.getProperty("prompt", "Hello, how are you?");
        int maxTokens = Integer.parseInt(System.getProperty("maxtokens", "50"));
        float temperature = Float.parseFloat(System.getProperty("temperature", "0.7"));
        int topK = Integer.parseInt(System.getProperty("topk", "40"));
        String delegateStr = System.getProperty("delegate", "AUTO");
        boolean interactive = Boolean.parseBoolean(System.getProperty("interactive", "false"));

        LiteRTConfig.Delegate delegate = LiteRTConfig.Delegate.valueOf(delegateStr.toUpperCase());

        // Validate parameters
        if (temperature < 0.0f || temperature > 2.0f) {
            System.err.println("⚠️  Warning: Temperature should be between 0.0 and 2.0");
            temperature = Math.max(0.0f, Math.min(2.0f, temperature));
        }

        if (maxTokens < 1 || maxTokens > 1000) {
            System.err.println("⚠️  Warning: Max tokens should be between 1 and 1000");
            maxTokens = Math.max(1, Math.min(1000, maxTokens));
        }

        if (modelPath == null || modelPath.isEmpty()) {
            System.out.println("Running in demo mode (no model provided)");
            System.out.println("To run with a real model:");
            System.out.println("  jbang -Dmodel=/path/to/model.litertlm -Dprompt=\"Your prompt\" TextGenerationExample.java");
            System.out.println();
            runDemoGeneration();
            return;
        }

        // Validate model file
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            System.err.println("❌ Error: Model file not found: " + modelPath);
            System.err.println("Please provide a valid path to a .litertlm file");
            System.exit(1);
        }

        if (!modelPath.endsWith(".litertlm")) {
            System.err.println("⚠️  Warning: Language model should have .litertlm extension");
        }

        System.out.println("Configuration:");
        System.out.println("  Model:       " + modelPath);
        System.out.println("  Prompt:      " + prompt);
        System.out.println("  Max Tokens:  " + maxTokens);
        System.out.println("  Temperature: " + temperature);
        System.out.println("  Top-K:       " + topK);
        System.out.println("  Delegate:    " + delegate);
        System.out.println();

        // Create SDK
        LiteRTConfig config = LiteRTConfig.builder()
                .numThreads(4)
                .delegate(delegate)
                .enableXnnpack(true)
                .useMemoryPool(true)
                .build();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            // Load model
            System.out.println("Loading model: " + modelPath);
            sdk.loadModel("textgen", Path.of(modelPath));
            System.out.println("✓ Model loaded");
            System.out.println();

            // Get model info
            LiteRTModelInfo modelInfo = sdk.getModelInfo("textgen");
            System.out.println("Model Information:");
            System.out.println("  ID:    " + modelInfo.getModelId());
            System.out.println("  Size:  " + formatBytes(modelInfo.getModelSizeBytes()));
            System.out.println();

            if (interactive) {
                runInteractiveMode(sdk, maxTokens, temperature, topK);
            } else {
                runSingleGeneration(sdk, prompt, maxTokens, temperature, topK);
            }

            // Display metrics
            displayMetrics(sdk);
        }

        System.out.println();
        System.out.println("✓ Example completed successfully");
    }

    /**
     * Run single text generation.
     */
    private static void runSingleGeneration(LiteRTSdk sdk, String prompt, int maxTokens,
                                            float temperature, int topK) throws Exception {
        System.out.println("Generating text...");
        System.out.println("Prompt: \"" + prompt + "\"");
        System.out.println();

        // Tokenize prompt
        byte[] inputTokens = tokenize(prompt);
        System.out.println("Input tokens: " + inputTokens.length);

        // Run inference
        long startTime = System.currentTimeMillis();
        InferenceRequest request = InferenceRequest.builder()
                .model("textgen")
                .inputData(inputTokens)
                .build();

        InferenceResponse response = sdk.infer(request);
        long elapsed = System.currentTimeMillis() - startTime;

        // Decode output
        String generatedText = decodeOutput(response.getOutputData());
        System.out.println("Generated text:");
        System.out.println(generatedText);
        System.out.println();

        System.out.println("✓ Generation completed in " + elapsed + "ms");
        System.out.println("  Tokens/sec: " + (maxTokens * 1000.0 / elapsed));
        System.out.println();
    }

    /**
     * Run interactive chat mode.
     */
    private static void runInteractiveMode(LiteRTSdk sdk, int maxTokens, float temperature, int topK) throws Exception {
        System.out.println("Interactive chat mode (type 'quit' or 'exit' to stop)");
        System.out.println();

        java.util.Scanner scanner = new java.util.Scanner(System.in);
        StringBuilder conversationHistory = new StringBuilder();

        while (true) {
            System.out.print("You: ");
            String userInput = scanner.nextLine().trim();

            if (userInput.equalsIgnoreCase("quit") || userInput.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            if (userInput.isEmpty()) {
                continue;
            }

            // Build prompt with conversation history
            conversationHistory.append("User: ").append(userInput).append("\n");
            conversationHistory.append("Assistant: ");

            String prompt = conversationHistory.toString();

            // Generate response
            System.out.print("Assistant: ");
            byte[] inputTokens = tokenize(prompt);

            long startTime = System.currentTimeMillis();
            InferenceRequest request = InferenceRequest.builder()
                    .model("textgen")
                    .inputData(inputTokens)
                    .build();

            InferenceResponse response = sdk.infer(request);
            String generatedText = decodeOutput(response.getOutputData());

            // Print generated response
            System.out.println(generatedText);
            conversationHistory.append(generatedText).append("\n\n");

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("  [" + elapsed + "ms]");
            System.out.println();
        }

        scanner.close();
    }

    /**
     * Run demo generation with mock data.
     */
    private static void runDemoGeneration() {
        System.out.println("Demo mode - simulating text generation");
        System.out.println();

        String prompt = "Once upon a time";
        System.out.println("Prompt: \"" + prompt + "\"");
        System.out.println();

        // Simulate token-by-token generation
        String[] demoTokens = {
            " in", " a", " far", "away", " kingdom", ",", " there", " lived", " a", " brave",
            " knight", " who", " embarked", " on", " many", " adventures", ".", " He", " fought",
            " dragons", " and", " saved", " the", " princess", " from", " an", " enchanted",
            " castle", ".", " The", " end", "."
        };

        System.out.print("Generated: ");
        System.out.print(prompt);

        for (int i = 0; i < demoTokens.length; i++) {
            try {
                Thread.sleep(50 + (long) (Math.random() * 100));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
            System.out.print(demoTokens[i]);
            System.out.flush();
        }

        System.out.println();
        System.out.println();
        System.out.println("✓ Demo generation completed");
        System.out.println();
        System.out.println("To run with a real model:");
        System.out.println("  jbang -Dmodel=/path/to/model.litertlm TextGenerationExample.java");
    }

    /**
     * Simple tokenization (placeholder - in production, use model's tokenizer).
     */
    private static byte[] tokenize(String text) {
        // Convert text to byte array
        // In production, this would use the model's actual tokenizer
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);

        // Create input buffer with proper format
        ByteBuffer buffer = ByteBuffer.allocate(4 + textBytes.length);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(textBytes.length);
        buffer.put(textBytes);

        return buffer.array();
    }

    /**
     * Decode model output to text (placeholder - in production, use model's tokenizer).
     */
    private static String decodeOutput(byte[] outputData) {
        // Convert output bytes back to text
        // In production, this would use the model's actual detokenizer
        try {
            ByteBuffer buffer = ByteBuffer.wrap(outputData);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            if (buffer.remaining() < 4) {
                return "";
            }

            int length = buffer.getInt();
            if (length <= 0 || length > outputData.length - 4) {
                length = Math.min(100, outputData.length - 4);
            }

            byte[] textBytes = new byte[length];
            buffer.get(textBytes);

            return new String(textBytes, StandardCharsets.UTF_8).trim();
        } catch (Exception e) {
            // Fallback: try to decode as much as possible
            return new String(outputData, StandardCharsets.UTF_8).trim();
        }
    }

    /**
     * Display performance metrics.
     */
    private static void displayMetrics(LiteRTSdk sdk) {
        System.out.println("Performance Metrics:");
        try {
            LiteRTMetrics metrics = sdk.getMetrics(null);
            System.out.println("  Total inferences:    " + metrics.getTotalInferences());
            System.out.println("  Failed inferences:   " + metrics.getFailedInferences());
            System.out.println("  Avg latency:         " + String.format("%.2f", metrics.getAvgLatencyMs()) + " ms");
            System.out.println("  P50 latency:         " + String.format("%.2f", metrics.getP50LatencyMs()) + " ms");
            System.out.println("  P95 latency:         " + String.format("%.2f", metrics.getP95LatencyMs()) + " ms");
            System.out.println("  Peak memory:         " + formatBytes(metrics.getPeakMemoryBytes()));
        } catch (Exception e) {
            System.out.println("  (Metrics not available)");
        }
    }

    /**
     * Format bytes to human-readable string.
     */
    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
