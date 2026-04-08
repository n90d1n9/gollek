///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
// Gollek LiteRT Standalone (.tflite) Example
// 
// Enhanced version with Metal diagnostics and --force-metal capability.
//
// DEPENDENCY RESOLUTION:
//   cd gollek && mvn clean install -pl plugins/runner/edge/litert/gollek-runner-litert -am -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-error-code:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
//DEPS org.slf4j:slf4j-simple:2.0.12
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

package tech.kayys.gollek.examples.standalone;

import tech.kayys.gollek.provider.litert.LiteRTNativeBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Robust standalone example for running .tflite models with hardware acceleration logs.
 */
public class LiteRTStandaloneExample {

    public static void main(String[] args) throws Exception {
        System.out.println("--- LiteRT Standalone Runner (Metal Verified) ---");

        String modelPath = null;
        String delegateStr = "CPU";
        boolean forceMetal = false;

        for (int i = 0; i < args.length; i++) {
            if ("--model".equals(args[i]) && i + 1 < args.length) {
                modelPath = args[++i];
            } else if ("--delegate".equals(args[i]) && i + 1 < args.length) {
                delegateStr = args[++i].toUpperCase();
            } else if ("--force-metal".equals(args[i])) {
                forceMetal = true;
                delegateStr = "GPU";
            }
        }

        if (modelPath == null && args.length > 0 && !args[0].startsWith("-")) {
            modelPath = args[0];
        }

        if (modelPath == null) {
            System.out.println("Usage: jbang LiteRTStandaloneExample.java --model <path> [--delegate GPU|CPU] [--force-metal]");
            System.exit(1);
        }

        Path p = Paths.get(modelPath);
        if (!java.nio.file.Files.exists(p)) {
            System.err.println("❌ Model not found: " + modelPath);
            System.exit(1);
        }

        // Native library diagnostics
        String home = System.getProperty("user.home");
        Path libPath = Paths.get(home, ".gollek", "libs", "libLiteRt.dylib");
        Path accelPath = Paths.get(home, ".gollek", "libs", "libLiteRtMetalAccelerator.dylib");
        Path customPath = Paths.get(home, ".gollek", "libs", "libgollek_metal.dylib");

        System.out.println("Platform Diagnostics:");
        System.out.println("  LiteRT Library:     " + (java.nio.file.Files.exists(libPath) ? "✓ Found" : "❌ Missing"));
        System.out.println("  Metal Accelerator:  " + (java.nio.file.Files.exists(accelPath) ? "✓ Found" : "⚠️  Missing (Official)"));
        System.out.println("  Gollek Bridge:      " + (java.nio.file.Files.exists(customPath) ? "✓ Found" : "ℹ️  Optional"));
        System.out.println();

        LiteRTNativeBindings bindings = new LiteRTNativeBindings(libPath);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment env = bindings.createEnvironment(arena);
            MemorySegment model = bindings.createModelFromFile(p.toString(), arena);
            MemorySegment opts = bindings.createOptions(arena);
            
            int accelFlag = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu;
            if ("GPU".equals(delegateStr) || "METAL".equals(delegateStr)) {
                accelFlag = LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
                System.out.println("🚀 Initializing GPU Acceleration (Force=" + forceMetal + ")...");
            } else {
                System.out.println("💻 Initializing CPU Inference...");
            }
            
            bindings.setOptionsHardwareAccelerators(opts, accelFlag); 

            try {
                long startTime = System.currentTimeMillis();
                MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
                long elapsed = System.currentTimeMillis() - startTime;
                
                System.out.println("✓ Compilation successful (" + elapsed + "ms)");
                System.out.println("  Hardware status: " + (bindings.isFullyAccelerated(compiledModel, arena) ? "ROCKING METAL 🤘" : "SOFT CPU ☁️"));

                // Signature Info
                int numSigs = bindings.getNumModelSignatures(model, arena);
                for (int i = 0; i < numSigs; i++) {
                    MemorySegment sig = bindings.getModelSignature(model, i, arena);
                    System.out.println("    - Sig " + i + ": " + bindings.getSignatureKey(sig, arena));
                }

                bindings.destroyCompiledModel(compiledModel);
            } catch (Exception e) {
                System.err.println("❌ Compilation Error: " + e.getMessage());
                if (delegateStr.equals("GPU")) {
                    System.err.println("   TIP: If Metal failed, ensure libLiteRtMetalAccelerator.dylib is CODESIGNED and in the libs folder.");
                    System.err.println("   OR try LiteRTMetalBridgeExample.java to use the native Gollek kernel.");
                }
            } finally {
                bindings.destroyOptions(opts);
                bindings.destroyModel(model);
                bindings.destroyEnvironment(env);
            }
        }

        System.out.println("\n✓ Execution finished.");
    }
}
