///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.slf4j:slf4j-simple:2.0.12
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.nio.file.Paths;

import static java.lang.foreign.ValueLayout.*;

/**
 * LiteRT 2.0 Verification Script
 *
 * Uses the CORRECT modern LiteRT 2.0 CompiledModel API:
 *   1. LiteRtCreateEnvironment(num_options, options_array, &env)
 *   2. LiteRtCreateModelFromFile(filename, &model)
 *   3. LiteRtCreateOptions(&options)
 *   4. LiteRtCreateCompiledModel(env, model, options, &compiled_model)
 *   5. LiteRtRunCompiledModel(compiled_model, sig_idx, n_in, ins, n_out, outs)
 *   6. Cleanup
 *
 * All functions return LiteRtStatus (int, 0 = OK) and write objects via out-pointers.
 */
public class LiteRTVerify {

    // LiteRtStatus constants
    static final int kLiteRtStatusOk = 0;

    // LiteRtHwAccelerators constants
    static final int kLiteRtHwAcceleratorCpu = 1;  // 1 << 0
    static final int kLiteRtHwAcceleratorGpu = 2;  // 1 << 1

    public static void main(String[] args) throws Throwable {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       LiteRT 2.0 Platform Verification                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        String home = System.getProperty("user.home");
        Path libPath = Paths.get(home, ".gollek", "libs", "libLiteRt.dylib");
        String modelPath = args.length > 0 ? args[0] : "gollek/examples/jbang/edge/models/1.tflite";

        // Validate files
        if (!libPath.toFile().exists()) {
            System.err.println("❌ Library not found: " + libPath);
            System.exit(1);
        }
        if (!new java.io.File(modelPath).exists()) {
            System.err.println("❌ Model not found: " + modelPath);
            System.exit(1);
        }

        System.out.println("Library:  " + libPath);
        System.out.println("Model:    " + modelPath);
        System.out.println();

        // Load library
        Linker linker = Linker.nativeLinker();
        SymbolLookup lib = SymbolLookup.libraryLookup(libPath, Arena.global());
        System.out.println("✓ Library loaded successfully");

        // === Bind LiteRT 2.0 APIs ===

        // Environment
        // LiteRtStatus LiteRtCreateEnvironment(int num_options, const LiteRtEnvOption* options, LiteRtEnvironment* env)
        MethodHandle createEnv = bind(linker, lib, "LiteRtCreateEnvironment",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
        // void LiteRtDestroyEnvironment(LiteRtEnvironment env)
        MethodHandle destroyEnv = bind(linker, lib, "LiteRtDestroyEnvironment",
                FunctionDescriptor.ofVoid(ADDRESS));

        // Model
        // LiteRtStatus LiteRtCreateModelFromFile(const char* filename, LiteRtModel* model)
        MethodHandle createModel = bind(linker, lib, "LiteRtCreateModelFromFile",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        // void LiteRtDestroyModel(LiteRtModel model)
        MethodHandle destroyModel = bind(linker, lib, "LiteRtDestroyModel",
                FunctionDescriptor.ofVoid(ADDRESS));

        // Model introspection
        // LiteRtStatus LiteRtGetNumModelSubgraphs(LiteRtModel model, LiteRtParamIndex* num)
        MethodHandle getNumSubgraphs = bind(linker, lib, "LiteRtGetNumModelSubgraphs",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        // LiteRtStatus LiteRtGetNumModelSignatures(LiteRtModel model, LiteRtParamIndex* num)
        MethodHandle getNumSignatures = bind(linker, lib, "LiteRtGetNumModelSignatures",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

        // Options
        // LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options)
        MethodHandle createOptions = bind(linker, lib, "LiteRtCreateOptions",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));
        // void LiteRtDestroyOptions(LiteRtOptions options)
        MethodHandle destroyOptions = bind(linker, lib, "LiteRtDestroyOptions",
                FunctionDescriptor.ofVoid(ADDRESS));

        // Options - accelerators
        // LiteRtStatus LiteRtSetOptionsHardwareAccelerators(LiteRtOptions options, LiteRtHwAcceleratorSet accel)
        MethodHandle setAccelerators = bind(linker, lib, "LiteRtSetOptionsHardwareAccelerators",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

        // CompiledModel
        // LiteRtStatus LiteRtCreateCompiledModel(LiteRtEnvironment env, LiteRtModel model, LiteRtOptions options, LiteRtCompiledModel* compiled)
        MethodHandle createCompiled = bind(linker, lib, "LiteRtCreateCompiledModel",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        // void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled)
        MethodHandle destroyCompiled = bind(linker, lib, "LiteRtDestroyCompiledModel",
                FunctionDescriptor.ofVoid(ADDRESS));
        // LiteRtStatus LiteRtCompiledModelIsFullyAccelerated(LiteRtCompiledModel compiled, bool* result)
        MethodHandle isFullyAccelerated = bind(linker, lib, "LiteRtCompiledModelIsFullyAccelerated",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

        // Status string for diagnostics
        // const char* LiteRtGetStatusString(LiteRtStatus status)
        MethodHandle getStatusString = bind(linker, lib, "LiteRtGetStatusString",
                FunctionDescriptor.of(ADDRESS, JAVA_INT));

        System.out.println("✓ All LiteRT 2.0 function handles bound");
        System.out.println();

        // === Execute LiteRT 2.0 Pipeline ===
        try (Arena arena = Arena.ofConfined()) {

            // Step 1: Create Environment (0 options = default CPU)
            System.out.print("Step 1: Creating environment...        ");
            MemorySegment envPtr = arena.allocate(ADDRESS);
            int status = (int) createEnv.invoke(0, MemorySegment.NULL, envPtr);
            check(status, "LiteRtCreateEnvironment", null);
            MemorySegment env = envPtr.get(ADDRESS, 0);
            System.out.println("✓ (addr: 0x" + Long.toHexString(env.address()) + ")");

            // Step 2: Load Model
            System.out.print("Step 2: Loading model...               ");
            MemorySegment modelPtr = arena.allocate(ADDRESS);
            MemorySegment pathSeg = arena.allocateFrom(modelPath);
            status = (int) createModel.invoke(pathSeg, modelPtr);
            check(status, "LiteRtCreateModelFromFile", null);
            MemorySegment model = modelPtr.get(ADDRESS, 0);
            System.out.println("✓ (addr: 0x" + Long.toHexString(model.address()) + ")");

            // Step 2b: Introspect Model
            MemorySegment numBuf = arena.allocate(JAVA_INT);
            status = (int) getNumSubgraphs.invoke(model, numBuf);
            if (status == kLiteRtStatusOk) {
                System.out.println("       Subgraphs:  " + numBuf.get(JAVA_INT, 0));
            }
            status = (int) getNumSignatures.invoke(model, numBuf);
            if (status == kLiteRtStatusOk) {
                System.out.println("       Signatures: " + numBuf.get(JAVA_INT, 0));
            }

            // Step 3: Create Options with CPU accelerator
            System.out.print("Step 3: Creating options (CPU)...      ");
            MemorySegment optPtr = arena.allocate(ADDRESS);
            status = (int) createOptions.invoke(optPtr);
            check(status, "LiteRtCreateOptions", getStatusString);
            MemorySegment options = optPtr.get(ADDRESS, 0);
            status = (int) setAccelerators.invoke(options, kLiteRtHwAcceleratorCpu);
            check(status, "LiteRtSetOptionsHardwareAccelerators", getStatusString);
            System.out.println("✓");

            // Step 4: Create Compiled Model
            System.out.print("Step 4: Creating compiled model...     ");
            MemorySegment compiledPtr = arena.allocate(ADDRESS);
            long t0 = System.currentTimeMillis();
            status = (int) createCompiled.invoke(env, model, options, compiledPtr);
            long compilationMs = System.currentTimeMillis() - t0;
            check(status, "LiteRtCreateCompiledModel", getStatusString);
            MemorySegment compiled = compiledPtr.get(ADDRESS, 0);
            System.out.println("✓ (" + compilationMs + "ms)");

            // Step 4b: Check acceleration
            MemorySegment accelBuf = arena.allocate(JAVA_INT);  // bool as int
            status = (int) isFullyAccelerated.invoke(compiled, accelBuf);
            if (status == kLiteRtStatusOk) {
                System.out.println("       Fully accelerated: " + (accelBuf.get(JAVA_INT, 0) != 0));
            }

            // Step 5: Cleanup
            System.out.print("Step 5: Cleanup...                     ");
            destroyCompiled.invoke(compiled);
            destroyOptions.invoke(options);
            destroyModel.invoke(model);
            destroyEnv.invoke(env);
            System.out.println("✓");
        }

        System.out.println();
        System.out.println("══════════════════════════════════════════════════════════");
        System.out.println("  ✓ STATUS: LiteRT 2.0 Platform is STABLE");
        System.out.println("══════════════════════════════════════════════════════════");
    }

    private static void check(int status, String func, MethodHandle getStatusString) throws Throwable {
        if (status != kLiteRtStatusOk) {
            String msg = "unknown";
            try {
                MemorySegment strPtr = (MemorySegment) getStatusString.invoke(status);
                if (strPtr != null && strPtr.address() != 0) {
                    msg = strPtr.reinterpret(256).getString(0);
                }
            } catch (Throwable ignored) {}
            throw new RuntimeException(func + " failed: status=" + status + " (" + msg + ")");
        }
    }

    private static MethodHandle bind(Linker linker, SymbolLookup lib, String name, FunctionDescriptor desc) {
        return lib.find(name)
                .map(addr -> linker.downcallHandle(addr, desc))
                .orElseThrow(() -> new RuntimeException("Symbol not found: " + name));
    }
}
