///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
// Gollek Gemma-4 Native Metal Chat Example
// 
// High-performance chat with Gemma-4 using the project's native Metal 
// engine. This script bypasses the failing LiteRT GPU delegate and uses
// zero-copy unified memory to run the transformer on the Apple M4 GPU.
//
// DEPENDENCIES:
//   cd gollek && mvn clean install -pl plugins/runner/edge/litert/gollek-runner-litert -am -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-kernel-metal:0.1.0-SNAPSHOT
//DEPS org.slf4j:slf4j-simple:2.0.12
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

package tech.kayys.gollek.examples.gemma;

import tech.kayys.gollek.provider.litert.LiteRTGemmaMetalRunner;
import tech.kayys.gollek.provider.litert.LiteRTTokenizer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;

/**
 * Interactive chat with Gemma-4 on Native Metal.
 */
public class LiteRTGemmaMetalExample {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       Gemma-4 Chat: Native Metal Acceleration           ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        String modelPathStr = null;
        for (int i = 0; i < args.length; i++) {
            if ("--model".equals(args[i]) && i + 1 < args.length) {
                modelPathStr = args[++i];
            }
        }

        if (modelPathStr == null && args.length > 0 && !args[0].startsWith("-")) {
            modelPathStr = args[0];
        }

        if (modelPathStr == null) {
            modelPathStr = System.getProperty("user.home") + "/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it_qualcomm_qcs8275.litertlm";
        }

        Path modelPath = Paths.get(modelPathStr);
        if (!java.nio.file.Files.exists(modelPath)) {
            System.err.println("❌ Model not found: " + modelPathStr);
            System.exit(1);
        }

        System.out.println("Model Path: " + modelPath.getFileName());

        // 1. Initialize Tokenizer
        System.out.print("Initializing Tokenizer... ");
        LiteRTTokenizer tokenizer = LiteRTTokenizer.create(modelPath);
        System.out.println("✓ (" + tokenizer.getVocabSize() + " tokens)");

        // 2. Initialize Gemma Native Metal Runner
        System.out.println("Initializing Gemma Native Metal Runner...");
        try (LiteRTGemmaMetalRunner runner = new LiteRTGemmaMetalRunner(modelPath, tokenizer)) {
            runner.initialize();

            System.out.println("\n✓ STATUS: Ready to Chat (Powered by Apple M4 Metal)");
            System.out.println("Enter your prompt (or 'exit' to quit):");

            try (Scanner scanner = new Scanner(System.in)) {
                while (true) {
                    System.out.print("\nYOU > ");
                    String prompt = scanner.nextLine();
                    if ("exit".equalsIgnoreCase(prompt) || "quit".equalsIgnoreCase(prompt)) break;

                    System.out.print("GEMMA > ");
                    long t0 = System.currentTimeMillis();
                    
                    runner.generate(prompt, token -> {
                        System.out.print(token);
                        System.out.flush();
                    });
                    
                    long elapsed = System.currentTimeMillis() - t0;
                    System.out.println("\n\n[Inference Time: " + elapsed + "ms]");
                }
            }
        } catch (Exception e) {
            System.err.println("\n❌ Runtime Error: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("\n✓ Session closed.");
    }
}
