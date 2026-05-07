package tech.kayys.gollek.backend.metal;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Native bridge to Metal kernels using Java FFM (Foreign Function & Memory) API.
 */
public final class NativeMetal {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;
    
    // Method Handles
    private static final MethodHandle INIT;
    private static final MethodHandle MATMUL;
    private static final MethodHandle ATTENTION;
    
    static {
        // Load the native library (assuming gollek-metal.dylib is in the library path)
        System.loadLibrary("gollek-metal");
        LOOKUP = SymbolLookup.loaderLookup();

        INIT = link("gollek_metal_init", FunctionDescriptor.of(ValueLayout.JAVA_INT));
        
        MATMUL = link("gollek_metal_matmul", FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        ATTENTION = link("gollek_metal_attention", FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, // out
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, // Q, K, V
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, // block_table, context_lens
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // B, T, H, D
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // block_size, max_blocks
            ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT // scale, is_causal, soft_cap
        ));
    }

    private static MethodHandle link(String name, FunctionDescriptor desc) {
        return LOOKUP.find(name)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() -> new RuntimeException("Failed to find native symbol: " + name));
    }

    public static int init() {
        try {
            return (int) INIT.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("Failed to initialize Metal", t);
        }
    }

    public static int matmul(MemorySegment C, MemorySegment A, MemorySegment B, 
                             int M, int K, int N, float alpha, float beta) {
        try {
            return (int) MATMUL.invokeExact(C, A, B, M, K, N, alpha, beta);
        } catch (Throwable t) {
            throw new RuntimeException("Metal MatMul failed", t);
        }
    }

    public static int attention(MemorySegment out, MemorySegment Q, MemorySegment K, MemorySegment V,
                                MemorySegment blockTable, MemorySegment contextLens,
                                int B, int T, int H, int D, int blockSize, int maxBlocks,
                                float scale, boolean causal, float softCap) {
        try {
            return (int) ATTENTION.invokeExact(out, Q, K, V, blockTable, contextLens, 
                B, T, H, D, blockSize, maxBlocks, scale, causal ? 1 : 0, softCap);
        } catch (Throwable t) {
            throw new RuntimeException("Metal Attention failed", t);
        }
    }
}
