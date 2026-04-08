///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek LiteRT .tflite Inference Example
// 
// Optimized for standard TensorFlow Lite (.tflite) models using LiteRT 2.0.
//
// DEPENDENCY:
//   cd gollek && mvn clean install -pl plugins/runner/edge/litert/gollek-runner-litert -am -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-error-code:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
//NOINTEGRATIONS
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.provider.litert.LiteRTNativeBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Super simple example for .tflite models (e.g. MobileNet, EfficientNet).
 */
public class LiteRTTfliteExample {

    public static void main(String[] args) throws Exception {
        System.out.println("--- LiteRT .tflite Standalone Runner ---");

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
            System.out.println("Usage: jbang LiteRTTfliteExample.java --model /path/to/model.tflite");
            System.exit(1);
        }

        Path p = Paths.get(modelPath);
        if (!java.nio.file.Files.exists(p)) {
            System.err.println("❌ Model file not found: " + modelPath);
            System.exit(1);
        }

        // Native library resolution
        String libPath = System.getProperty("LITERT_LIBRARY_PATH", "/Users/bhangun/.gollek/libs/libLiteRt.dylib");
        System.out.println("Using LiteRT Library: " + libPath);
        LiteRTNativeBindings bindings = new LiteRTNativeBindings(Paths.get(libPath));

        try (Arena arena = Arena.ofConfined()) {
            // 1. Initialize Runtime Environment
            MemorySegment env = bindings.createEnvironment(arena);
            
            // 2. Load Model
            MemorySegment model = bindings.createModelFromFile(p.toString(), arena);
            
            // 3. Set Hardware Options (CPU by default)
            MemorySegment opts = bindings.createOptions(arena);
            bindings.setOptionsHardwareAccelerators(opts, LiteRTNativeBindings.kLiteRtHwAcceleratorCpu); 

            // 4. Compile Model
            MemorySegment compiledModel = bindings.createCompiledModel(env, model, opts, arena);
            
            System.out.println("✓ Model Compiled: " + p.getFileName());
            System.out.println("  Acceleration: " + (bindings.isFullyAccelerated(compiledModel, arena) ? "GPU/NPU" : "CPU"));

            // 5. Introspect Signatures
            int numSigs = bindings.getNumModelSignatures(model, arena);
            System.out.println("  Signatures: " + numSigs);
            
            for (int i = 0; i < numSigs; i++) {
                MemorySegment sig = bindings.getModelSignature(model, i, arena);
                String key = bindings.getSignatureKey(sig, arena);
                int inputs = bindings.getNumSignatureInputs(sig, arena);
                int outputs = bindings.getNumSignatureOutputs(sig, arena);
                System.out.println("    [" + key + "] " + inputs + " inputs -> " + outputs + " outputs");
            }

            // Cleanup
            bindings.destroyCompiledModel(compiledModel);
            bindings.destroyOptions(opts);
            bindings.destroyModel(model);
            bindings.destroyEnvironment(env);
        }

        System.out.println("\n✓ Execution finished successfully.");
    }
}
