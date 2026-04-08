///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
// Gollek LiteRT-LM (.litertlm) Gemma Example
// 
// Specialized for large language models in the LiteRT container format (.litertlm).
//
// DEPENDENCY:
//   cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS org.slf4j:slf4j-simple:2.0.12
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Specialized example for .litertlm models (Gemma-4, etc).
 */
public class LiteRTGemmaExample {

    public static void main(String[] args) throws Exception {
        System.out.println("--- LiteRT-LM (.litertlm) Gemma Runner ---");

        String modelPath = null;
        for (int i = 0; i < args.length; i++) {
            if ("--model".equals(args[i]) && i + 1 < args.length) {
                modelPath = args[i+1];
                break;
            }
        }

        if (modelPath == null && args.length > 0 && !args[0].startsWith("-")) {
            modelPath = args[0];
        }

        if (modelPath == null) {
            System.out.println("Usage: jbang LiteRTGemmaExample.java --model <path_to_gemma.litertlm>");
            System.exit(1);
        }

        Path p = Paths.get(modelPath);
        if (!java.nio.file.Files.exists(p)) {
            System.err.println("❌ Model file not found: " + modelPath);
            System.exit(1);
        }

        System.out.println("Model Identified: " + p.getFileName());

        LiteRTConfig config = LiteRTConfig.builder()
                .delegate(LiteRTConfig.Delegate.CPU)
                .numThreads(8)
                .build();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            System.out.println("✓ SDK Initialized");

            try {
                System.out.println("Loading language model...");
                sdk.loadModel("gemma", p);
                
                System.out.println("Generating text...");
                var response = sdk.infer(InferenceRequest.builder()
                        .model("gemma")
                        .message(Message.user("Explain quantum computing in one sentence."))
                        .build());
                        
                System.out.println("Response: " + response.getContent());
            } catch (Exception e) {
                System.err.println("\n❌ LLM Runtime Error: " + e.getMessage());
                if (e.getMessage() != null && e.getMessage().contains("LiteRT-LM")) {
                    System.err.println("\n[GUIDE] This gemma model uses the LiteRT-LM container.");
                    System.err.println("   Standard LiteRT inference is for vision/generic tensors.");
                    System.err.println("   LLM generation requires the libLiteRtLm.dylib runtime library.");
                }
            }
        }

        System.out.println("\n✓ Process finished.");
    }
}
