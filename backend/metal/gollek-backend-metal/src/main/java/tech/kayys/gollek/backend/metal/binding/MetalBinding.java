package tech.kayys.gollek.metal.binding;

import org.jboss.logging.Logger;

import java.io.File;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM-based binding to the Gollek Metal bridge ({@code libgollek_metal.dylib}).
 *
 * <p>
 * Mirrors the pattern of the existing {@code FlashAttention3Binding} and
 * {@code PagedAttentionBinding}: a singleton loaded once at startup via
 * {@link SymbolLookup#libraryLookup}, falling back to a CPU implementation
 * when the dylib is absent (e.g., on non-Apple hardware).
 *
 * <h2>Unified Memory optimisation</h2>
 * <p>
 * On Apple Silicon the CPU and GPU share the same physical DRAM.
 * A {@link MemorySegment} allocated with {@link Arena#ofShared()} has a CPU
 * virtual address that Metal can access directly as a
 * {@code MTLStorageModeShared} buffer. The bridge's {@code wrap_ptr()}
 * function (in {@code gollek_metal_bridge.m}) calls
 * {@code newBufferWithBytesNoCopy} to hand the existing pointer to Metal —
 * <b>zero copy, zero extra allocation</b>.
 *
 * <h2>Lifecycle</h2>
 * 
 * <pre>{@code
 * // At startup (e.g., MetalRunner.initialize()):
 * boolean loaded = MetalBinding.initialize(
 *         Path.of("/opt/gollek/lib/libgollek_metal.dylib"));
 * MetalBinding binding = MetalBinding.getInstance();
 *
 * // Use:
 * binding.init();
 * binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
 * binding.attention(out, Q, K, V, bt, ctx, B, T, H, D, blockSize, maxBlocks, scale, 1);
 * }</pre>
 */
public class MetalBinding {

    private static final Logger LOG = Logger.getLogger(MetalBinding.class);
    private static volatile MetalBinding instance;

    // ── Function names ────────────────────────────────────────────────────────

    private static final String FN_INIT = "gollek_metal_init";
    private static final String FN_AVAIL_MEM = "gollek_metal_available_memory";
    private static final String FN_ALLOC = "gollek_metal_alloc";
    private static final String FN_MATMUL = "gollek_metal_matmul";
    private static final String FN_MATMUL_TB = "gollek_metal_matmul_tb";
    private static final String FN_MATMUL_TB_HALF = "gollek_metal_matmul_tb_half";
    private static final String FN_MATMUL_TB_HALF_PAIR = "gollek_metal_matmul_tb_half_pair";
    private static final String FN_MATMUL_TB_HALF_PAIR_MIXED = "gollek_metal_matmul_tb_half_pair_mixed";
    private static final String FN_MATMUL_TB_HALF_TRIPLE_MIXED = "gollek_metal_matmul_tb_half_triple_mixed";
    private static final String FN_SWIGLU_FFN_HALF = "gollek_metal_swiglu_ffn_half";
    private static final String FN_GEGLU_FFN_HALF = "gollek_metal_geglu_ffn_half";
    private static final String FN_MATVEC_TB_HALF = "gollek_metal_matvec_tb_half";
    private static final String FN_MATVEC_T_HALF = "gollek_metal_matvec_t_half";
    private static final String FN_MATVEC_TB_HALF_PAIR = "gollek_metal_matvec_tb_half_pair";
    private static final String FN_MATVEC_TB_HALF_TRIPLE_MIXED = "gollek_metal_matvec_tb_half_triple_mixed";
    private static final String FN_ATTENTION = "gollek_metal_attention";
    private static final String FN_ATTENTION_GQA = "gollek_metal_attention_gqa";
    private static final String FN_ATTENTION_WINDOWED = "gollek_metal_attention_windowed";
    private static final String FN_ATTENTION_GQA_WINDOWED = "gollek_metal_attention_gqa_windowed";
    private static final String FN_RMSNORM = "gollek_metal_rmsnorm";
    private static final String FN_SILU_FFN = "gollek_metal_silu_ffn";
    private static final String FN_GELU_FFN = "gollek_metal_gelu_ffn";
    private static final String FN_SET_MPS_MATVEC_ENABLED = "gollek_metal_set_mps_matvec_enabled";
    private static final String FN_SET_MPS_MATVEC_AUTOTUNE_ENABLED = "gollek_metal_set_mps_matvec_autotune_enabled";
    private static final String FN_SET_MPS_MATVEC_MAX_INNER = "gollek_metal_set_mps_matvec_max_inner";
    private static final String FN_SET_MPS_MATVEC_MAX_OUTPUT = "gollek_metal_set_mps_matvec_max_output";
    private static final String FN_SET_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT = "gollek_metal_set_mps_matvec_autotune_max_output";
    private static final String FN_UNIFIED_MEM = "gollek_metal_is_unified_memory";
    private static final String FN_DEVICE_NAME = "gollek_metal_device_name";

    // Basic Math Operations
    private static final String FN_ADD = "gollek_metal_add";
    private static final String FN_SUB = "gollek_metal_sub";
    private static final String FN_MUL = "gollek_metal_mul";
    private static final String FN_DIV = "gollek_metal_div";
    private static final String FN_RELU = "gollek_metal_relu";
    private static final String FN_SIGMOID = "gollek_metal_sigmoid";
    private static final String FN_TANH = "gollek_metal_tanh";
    private static final String FN_EXP = "gollek_metal_exp";
    private static final String FN_LOG = "gollek_metal_log";
    private static final String FN_SUM = "gollek_metal_sum";
    private static final String FN_MEAN = "gollek_metal_mean";
    private static final String FN_POW = "gollek_metal_pow";
    private static final String FN_TRANSPOSE2D = "gollek_metal_transpose2d";
    private static final String ENABLE_ELEMENTWISE_KERNELS_PROPERTY = "gollek.metal.enable_elementwise_kernels";
    private static final String ENABLE_ELEMENTWISE_KERNELS_ENV = "GOLLEK_METAL_ENABLE_ELEMENTWISE_KERNELS";
    private static final String ENABLE_MPS_MATVEC_PROPERTY = "gollek.metal.enable_mps_matvec";
    private static final String ENABLE_MPS_MATVEC_AUTOTUNE_PROPERTY = "gollek.metal.mps_matvec.autotune";
    private static final String MPS_MATVEC_MAX_INNER_PROPERTY = "gollek.metal.mps_matvec.max_inner";
    private static final String MPS_MATVEC_MAX_OUTPUT_PROPERTY = "gollek.metal.mps_matvec.max_output";
    private static final String MPS_MATVEC_AUTOTUNE_MAX_OUTPUT_PROPERTY = "gollek.metal.mps_matvec.autotune.max_output";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> handles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;
    private volatile boolean runtimeCpuFallback;

    private MetalBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable)
            bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /**
     * Automatically discover and initialize the Metal native bridge.
     * Searches in standard locations and java.library.path.
     *
     * @return true if successfully loaded.
     */
    public static boolean initialize() {
        return initialize(findLibrary());
    }

    public static boolean initialize(Path libraryPath) {
        if (isNativeBuildTime()) {
            LOG.info("MetalBinding: skipping initialization during native-image build time");
            return false;
        }
        if (instance != null)
            return instance.nativeAvailable;
        if (libraryPath == null || !Files.exists(libraryPath)) {
            LOG.warnf("MetalBinding: dylib not found at %s — CPU fallback active", libraryPath);
            instance = new MetalBinding(null);
            return false;
        }
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new MetalBinding(lk);
            LOG.infof("MetalBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("MetalBinding: failed to load %s (%s) — CPU fallback active",
                    libraryPath, e.getMessage());
            instance = new MetalBinding(null);
            return false;
        }
    }

    private static boolean isNativeBuildTime() {
        return "buildtime".equals(System.getProperty("org.graalvm.nativeimage.imagecode"));
    }

    private static Path findLibrary() {
        return MetalLibraryDiscovery.findLibrary();
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new MetalBinding(null);
        LOG.info("MetalBinding: CPU fallback mode");
    }

    public static MetalBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException(
                    "MetalBinding not initialised — call initialize() first");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialise the Metal device and command queue on the native side.
     * 
     * @return 0 on success, negative on error.
     */
    public int init() {
        if (useCpuFallback())
            return 0; // CPU fallback — nothing to init
        int rc = (int) invoke(FN_INIT);
        if (rc != 0) {
            runtimeCpuFallback = true;
            LOG.warnf("MetalBinding: init failed with code %d — CPU fallback active", rc);
            return 0;
        }
        configureMpsMatvecFromProperties();
        return 0;
    }

    private void configureMpsMatvecFromProperties() {
        if (!nativeAvailable || useCpuFallback()) {
            return;
        }
        if (handles.containsKey(FN_SET_MPS_MATVEC_ENABLED)) {
            setMpsMatvecEnabled(Boolean.getBoolean(ENABLE_MPS_MATVEC_PROPERTY));
        }
        if (handles.containsKey(FN_SET_MPS_MATVEC_AUTOTUNE_ENABLED)) {
            setMpsMatvecAutotuneEnabled(Boolean.getBoolean(ENABLE_MPS_MATVEC_AUTOTUNE_PROPERTY));
        }
        if (handles.containsKey(FN_SET_MPS_MATVEC_MAX_INNER)) {
            Integer parsed = parseOptionalIntegerProperty(MPS_MATVEC_MAX_INNER_PROPERTY);
            if (parsed != null) setMpsMatvecMaxInner(parsed);
        }
        if (handles.containsKey(FN_SET_MPS_MATVEC_MAX_OUTPUT)) {
            Integer parsed = parseOptionalIntegerProperty(MPS_MATVEC_MAX_OUTPUT_PROPERTY);
            if (parsed != null) setMpsMatvecMaxOutput(parsed);
        }
        if (handles.containsKey(FN_SET_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT)) {
            Integer parsed = parseOptionalIntegerProperty(MPS_MATVEC_AUTOTUNE_MAX_OUTPUT_PROPERTY);
            if (parsed != null) setMpsMatvecAutotuneMaxOutput(parsed);
        }
    }

    private Integer parseOptionalIntegerProperty(String property) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            LOG.warnf("MetalBinding: ignoring invalid %s=%s", property, raw);
            return null;
        }
    }

    public void setMpsMatvecEnabled(boolean enabled) {
        if (useCpuFallback() || !handles.containsKey(FN_SET_MPS_MATVEC_ENABLED)) {
            return;
        }
        invoke(FN_SET_MPS_MATVEC_ENABLED, enabled ? 1 : 0);
    }

    public void setMpsMatvecAutotuneEnabled(boolean enabled) {
        if (useCpuFallback() || !handles.containsKey(FN_SET_MPS_MATVEC_AUTOTUNE_ENABLED)) {
            return;
        }
        invoke(FN_SET_MPS_MATVEC_AUTOTUNE_ENABLED, enabled ? 1 : 0);
    }

    public void setMpsMatvecMaxInner(int maxInner) {
        if (useCpuFallback() || !handles.containsKey(FN_SET_MPS_MATVEC_MAX_INNER)) {
            return;
        }
        invoke(FN_SET_MPS_MATVEC_MAX_INNER, maxInner);
    }

    public void setMpsMatvecMaxOutput(int maxOutput) {
        if (useCpuFallback() || !handles.containsKey(FN_SET_MPS_MATVEC_MAX_OUTPUT)) {
            return;
        }
        invoke(FN_SET_MPS_MATVEC_MAX_OUTPUT, maxOutput);
    }

    public void setMpsMatvecAutotuneMaxOutput(int maxOutput) {
        if (useCpuFallback() || !handles.containsKey(FN_SET_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT)) {
            return;
        }
        invoke(FN_SET_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT, maxOutput);
    }

    /**
     * Available GPU memory in bytes (shared DRAM on Apple Silicon).
     */
    public long availableMemory() {
        if (useCpuFallback())
            return Runtime.getRuntime().freeMemory();
        return (long) invoke(FN_AVAIL_MEM);
    }

    /**
     * Whether the device uses unified memory (always true on Apple Silicon
     * M-series).
     */
    public boolean isUnifiedMemory() {
        if (useCpuFallback())
            return false;
        return ((int) invoke(FN_UNIFIED_MEM)) == 1;
    }

    /**
     * Metal device display name (e.g., "Apple M3 Pro").
     */
    public String deviceName() {
        if (useCpuFallback())
            return "CPU (no Metal)";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(256L);
            invoke(FN_DEVICE_NAME, buf, 256);
            return buf.getString(0, java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    /**
     * Matrix multiplication via MPS: C = alpha · A × B + beta · C
     *
     * <p>
     * On Apple Silicon this dispatches to the AMX blocks via
     * MPSMatrixMultiplication — significantly faster than BLAS on CPU.
     *
     * @param C     Output [M × N float32]
     * @param A     Left [M × K float32]
     * @param B     Right [K × N float32]
     * @param M,K,N Dimensions
     * @param alpha Scale factor
     * @param beta  Accumulation (0 = overwrite)
     * @return 0 on success
     */
    public int matmul(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta) {
        if (useCpuFallback()) {
            return MetalCpuFallback.matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL, C, A, B, M, K, N, alpha, beta);
    }

    public int matmulTransposedRight(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta) {
        if (useCpuFallback()) {
            return MetalCpuFallback.matmulTransposedRight(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL_TB, C, A, B, M, K, N, alpha, beta);
    }

    public int matmulTransposedRightHalf(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta, boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matmulTransposedRightHalf not supported in CPU fallback");
        }
        return (int) invoke(FN_MATMUL_TB_HALF, C, A, B, M, K, N, alpha, beta, isBf16 ? 1 : 0);
    }

    public boolean supportsMatmulTransposedRightHalfPair() {
        return !useCpuFallback() && handles.containsKey(FN_MATMUL_TB_HALF_PAIR);
    }

    public int matmulTransposedRightHalfPair(MemorySegment C0, MemorySegment C1,
            MemorySegment A,
            MemorySegment B0, MemorySegment B1,
            int M, int K, int N, float alpha, float beta, boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matmulTransposedRightHalfPair not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATMUL_TB_HALF_PAIR)) {
            return -2;
        }
        return (int) invoke(FN_MATMUL_TB_HALF_PAIR, C0, C1, A, B0, B1, M, K, N, alpha, beta, isBf16 ? 1 : 0);
    }

    public boolean supportsMatmulTransposedRightHalfPairMixed() {
        return !useCpuFallback() && handles.containsKey(FN_MATMUL_TB_HALF_PAIR_MIXED);
    }

    public int matmulTransposedRightHalfPairMixed(MemorySegment C0, MemorySegment C1,
            MemorySegment A,
            MemorySegment B0, MemorySegment B1,
            int M, int K, int N0, int N1, float alpha, float beta, boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matmulTransposedRightHalfPairMixed not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATMUL_TB_HALF_PAIR_MIXED)) {
            return -2;
        }
        return (int) invoke(FN_MATMUL_TB_HALF_PAIR_MIXED, C0, C1, A, B0, B1,
                M, K, N0, N1, alpha, beta, isBf16 ? 1 : 0);
    }

    public boolean supportsMatmulTransposedRightHalfTripleMixed() {
        return !useCpuFallback() && handles.containsKey(FN_MATMUL_TB_HALF_TRIPLE_MIXED);
    }

    public int matmulTransposedRightHalfTripleMixed(MemorySegment C0, MemorySegment C1, MemorySegment C2,
            MemorySegment A,
            MemorySegment B0, MemorySegment B1, MemorySegment B2,
            int M, int K, int N0, int N1, int N2, float alpha, float beta, boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matmulTransposedRightHalfTripleMixed not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATMUL_TB_HALF_TRIPLE_MIXED)) {
            return -2;
        }
        return (int) invoke(FN_MATMUL_TB_HALF_TRIPLE_MIXED, C0, C1, C2, A, B0, B1, B2,
                M, K, N0, N1, N2, alpha, beta, isBf16 ? 1 : 0);
    }

    public boolean supportsSwigluFfnHalf() {
        return !useCpuFallback() && handles.containsKey(FN_SWIGLU_FFN_HALF);
    }

    public boolean supportsGegluFfnHalf() {
        return !useCpuFallback() && handles.containsKey(FN_GEGLU_FFN_HALF);
    }

    public int swigluFfnHalf(MemorySegment C,
            MemorySegment A,
            MemorySegment gateW,
            MemorySegment upW,
            MemorySegment downW,
            int M,
            int inputDim,
            int intermediateDim,
            int outputDim,
            boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal swigluFfnHalf not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_SWIGLU_FFN_HALF)) {
            return -2;
        }
        return (int) invoke(FN_SWIGLU_FFN_HALF, C, A, gateW, upW, downW,
                M, inputDim, intermediateDim, outputDim, isBf16 ? 1 : 0);
    }

    public int gegluFfnHalf(MemorySegment C,
            MemorySegment A,
            MemorySegment gateW,
            MemorySegment upW,
            MemorySegment downW,
            int M,
            int inputDim,
            int intermediateDim,
            int outputDim,
            boolean isBf16) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal gegluFfnHalf not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_GEGLU_FFN_HALF)) {
            return -2;
        }
        return (int) invoke(FN_GEGLU_FFN_HALF, C, A, gateW, upW, downW,
                M, inputDim, intermediateDim, outputDim, isBf16 ? 1 : 0);
    }

    public boolean supportsMatvecTransposedRightHalfPair() {
        return !useCpuFallback() && handles.containsKey(FN_MATVEC_TB_HALF_PAIR);
    }

    public boolean supportsMatvecTransposedRightHalfTripleMixed() {
        return !useCpuFallback() && handles.containsKey(FN_MATVEC_TB_HALF_TRIPLE_MIXED);
    }

    public boolean supportsMatvecTransposedRightHalf() {
        return !useCpuFallback() && handles.containsKey(FN_MATVEC_TB_HALF);
    }

    public boolean supportsMatvecTransposedWeightHalf() {
        return !useCpuFallback() && handles.containsKey(FN_MATVEC_T_HALF);
    }

    public int matvecTransposedRightHalf(MemorySegment C,
            MemorySegment A,
            MemorySegment B,
            int K, int N) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matvecTransposedRightHalf not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATVEC_TB_HALF)) {
            return -2;
        }
        return (int) invoke(FN_MATVEC_TB_HALF, C, A, B, K, N);
    }

    public int matvecTransposedWeightHalf(MemorySegment C,
            MemorySegment A,
            MemorySegment B,
            int K, int N) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matvecTransposedWeightHalf not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATVEC_T_HALF)) {
            return -2;
        }
        return (int) invoke(FN_MATVEC_T_HALF, C, A, B, K, N);
    }

    public int matvecTransposedRightHalfPair(MemorySegment C0, MemorySegment C1,
            MemorySegment A,
            MemorySegment B0, MemorySegment B1,
            int K, int N) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matvecTransposedRightHalfPair not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATVEC_TB_HALF_PAIR)) {
            return -2;
        }
        return (int) invoke(FN_MATVEC_TB_HALF_PAIR, C0, C1, A, B0, B1, K, N);
    }

    public int matvecTransposedRightHalfTripleMixed(MemorySegment C0, MemorySegment C1, MemorySegment C2,
            MemorySegment A,
            MemorySegment B0, MemorySegment B1, MemorySegment B2,
            int K, int N0, int N1, int N2) {
        if (useCpuFallback()) {
            throw new UnsupportedOperationException("Metal matvecTransposedRightHalfTripleMixed not supported in CPU fallback");
        }
        if (!handles.containsKey(FN_MATVEC_TB_HALF_TRIPLE_MIXED)) {
            return -2;
        }
        return (int) invoke(FN_MATVEC_TB_HALF_TRIPLE_MIXED, C0, C1, C2, A, B0, B1, B2,
                K, N0, N1, N2);
    }

    /**
     * Paged softmax attention via Metal MPS.
     *
     * <p>
     * K/V cache is accessed directly from Gollek's
     * {@link tech.kayys.gollek.kvcache.PhysicalBlockPool} off-heap slabs —
     * no copy on Apple Silicon because the slabs are already in shared DRAM.
     *
     * @param out         [B, T, H, D] float32
     * @param Q           [B, T, H, D] float32
     * @param K_cache     paged K pool (from PhysicalBlockPool.rawKPool())
     * @param V_cache     paged V pool (from PhysicalBlockPool.rawVPool())
     * @param blockTable  [B, maxBlocks] int32
     * @param contextLens [B] int32
     * @param scale       1/sqrt(D)
     * @param isCausal    1 = causal mask
     * @return 0 on success
     */
    public int attention(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, float softCap) {
        return attentionWindowed(out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, H, D, blockSize, maxBlocks, scale, isCausal, 0, 0, softCap);
    }

    public int attentionWindowed(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int Hkv, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, int queryStartPos, int slidingWindow, float softCap) {
        if (useCpuFallback()) {
            return MetalCpuFallback.attentionWindowed(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal,
                    queryStartPos, slidingWindow, softCap);
        }
        MethodHandle mh = handles.get(FN_ATTENTION_WINDOWED);
        if (mh == null) {
            if (queryStartPos == 0 && slidingWindow <= 0 && Hkv == H) {
                return (int) invoke(FN_ATTENTION,
                        out, Q, K_cache, V_cache, blockTable, contextLens,
                        B, T, H, D, blockSize, maxBlocks, scale, isCausal, softCap);
            }
            return MetalCpuFallback.attentionWindowed(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal,
                    queryStartPos, slidingWindow, softCap);
        }
        return (int) invoke(FN_ATTENTION_WINDOWED,
                out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, D, blockSize, maxBlocks, scale, isCausal,
                queryStartPos, slidingWindow, softCap);
    }

    public int attentionGqa(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int Hkv, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, float softCap) {
        return attentionGqaWindowed(out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal, 0, 0, softCap);
    }

    public int attentionGqaWindowed(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int Hkv, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, int queryStartPos, int slidingWindow, float softCap) {
        if (useCpuFallback()) {
            return MetalCpuFallback.attentionWindowed(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal,
                    queryStartPos, slidingWindow, softCap);
        }
        MethodHandle mh = handles.get(FN_ATTENTION_GQA_WINDOWED);
        if (mh == null) {
            if (queryStartPos == 0 && slidingWindow <= 0) {
                MethodHandle legacyGqa = handles.get(FN_ATTENTION_GQA);
                if (legacyGqa != null) {
                    return (int) invoke(FN_ATTENTION_GQA,
                            out, Q, K_cache, V_cache, blockTable, contextLens,
                            B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal, softCap);
                }
            }
            return MetalCpuFallback.attentionWindowed(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal,
                    queryStartPos, slidingWindow, softCap);
        }
        return (int) invoke(FN_ATTENTION_GQA_WINDOWED,
                out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal,
                queryStartPos, slidingWindow, softCap);
    }

    public boolean isWindowedAttentionAvailable() {
        return handles.containsKey(FN_ATTENTION_WINDOWED);
    }

    /**
     * RMS normalisation: out = x / rms(x) * weight.
     * 
     * @return 0 on success
     */
    public int rmsNorm(MemorySegment out, MemorySegment x,
            MemorySegment weight, int N, float eps, boolean addOne) {
        if (useCpuFallback())
            return MetalCpuFallback.rmsNorm(out, x, weight, N, eps, addOne);
        return (int) invoke(FN_RMSNORM, out, x, weight, N, eps, addOne ? 1 : 0);
    }

    /**
     * SiLU-gated FFN: out = silu(gate) * up
     * 
     * @return 0 on success
     */
    public int siluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        if (useCpuFallback())
            return MetalCpuFallback.siluFfn(out, gate, up, N);
        return (int) invoke(FN_SILU_FFN, out, gate, up, N);
    }

    /**
     * GeLU-gated FFN: out = gelu(gate) * up
     *
     * @return 0 on success
     */
    public int geluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_GELU_FFN))
            return MetalCpuFallback.geluFfn(out, gate, up, N);
        return (int) invoke(FN_GELU_FFN, out, gate, up, N);
    }

    // ── Basic Math API ────────────────────────────────────────────────────────

    public int add(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_ADD) || !nativeElementwiseKernelsEnabled())
            return MetalCpuFallback.add(C, A, B, N);
        return (int) invoke(FN_ADD, C, A, B, N);
    }

    public int sub(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_SUB))
            return MetalCpuFallback.sub(C, A, B, N);
        return (int) invoke(FN_SUB, C, A, B, N);
    }

    public int mul(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_MUL))
            return MetalCpuFallback.mul(C, A, B, N);
        return (int) invoke(FN_MUL, C, A, B, N);
    }

    public int div(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_DIV))
            return MetalCpuFallback.div(C, A, B, N);
        return (int) invoke(FN_DIV, C, A, B, N);
    }

    public int relu(MemorySegment C, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_RELU))
            return MetalCpuFallback.relu(C, A, N);
        return (int) invoke(FN_RELU, C, A, N);
    }

    public int sigmoid(MemorySegment C, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_SIGMOID))
            return MetalCpuFallback.sigmoid(C, A, N);
        return (int) invoke(FN_SIGMOID, C, A, N);
    }

    public int tanh(MemorySegment C, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_TANH))
            return MetalCpuFallback.tanh(C, A, N);
        return (int) invoke(FN_TANH, C, A, N);
    }

    public int exp(MemorySegment C, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_EXP))
            return MetalCpuFallback.exp(C, A, N);
        return (int) invoke(FN_EXP, C, A, N);
    }

    public int log(MemorySegment C, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_LOG))
            return MetalCpuFallback.log(C, A, N);
        return (int) invoke(FN_LOG, C, A, N);
    }

    public int sum(MemorySegment out, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_SUM))
            return MetalCpuFallback.sum(out, A, N);
        return (int) invoke(FN_SUM, out, A, N);
    }

    public int mean(MemorySegment out, MemorySegment A, int N) {
        if (useCpuFallback() || !handles.containsKey(FN_MEAN))
            return MetalCpuFallback.mean(out, A, N);
        return (int) invoke(FN_MEAN, out, A, N);
    }

    public int pow(MemorySegment C, MemorySegment A, int N, float p) {
        if (useCpuFallback() || !handles.containsKey(FN_POW))
            return MetalCpuFallback.pow(C, A, N, p);
        return (int) invoke(FN_POW, C, A, N, p);
    }

    public int transpose2d(MemorySegment C, MemorySegment A, int rows, int cols) {
        if (useCpuFallback() || !handles.containsKey(FN_TRANSPOSE2D))
            return MetalCpuFallback.transpose2d(C, A, rows, cols);
        return (int) invoke(FN_TRANSPOSE2D, C, A, rows, cols);
    }

    private boolean useCpuFallback() {
        return !nativeAvailable || runtimeCpuFallback;
    }

    private static boolean nativeElementwiseKernelsEnabled() {
        String explicit = System.getProperty(ENABLE_ELEMENTWISE_KERNELS_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        String env = System.getenv(ENABLE_ELEMENTWISE_KERNELS_ENV);
        return env != null
                && ("1".equals(env)
                || "true".equalsIgnoreCase(env)
                || "yes".equalsIgnoreCase(env));
    }

    // ── FFM binding ───────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_metal_init()
        bind(FN_INIT, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // long gollek_metal_available_memory()
        bind(FN_AVAIL_MEM, FunctionDescriptor.of(ValueLayout.JAVA_LONG));

        // void* gollek_metal_alloc(size_t bytes, size_t align)
        bind(FN_ALLOC, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        // int gollek_metal_matmul(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        bind(FN_MATMUL_TB, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        bind(FN_MATMUL_TB_HALF, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        bind(FN_MATMUL_TB_HALF_PAIR, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        bind(FN_MATMUL_TB_HALF_PAIR_MIXED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        bind(FN_MATMUL_TB_HALF_TRIPLE_MIXED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        bind(FN_SWIGLU_FFN_HALF, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        bind(FN_GEGLU_FFN_HALF, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        bind(FN_MATVEC_TB_HALF, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        bind(FN_MATVEC_T_HALF, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        bind(FN_MATVEC_TB_HALF_PAIR, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        bind(FN_MATVEC_TB_HALF_TRIPLE_MIXED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // causal, soft_cap)
        bind(FN_ATTENTION, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        bind(FN_ATTENTION_GQA, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        bind(FN_ATTENTION_WINDOWED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        bind(FN_ATTENTION_GQA_WINDOWED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        // int gollek_metal_rmsnorm(out, x, weight, N, eps, add_one)
        bind(FN_RMSNORM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // int gollek_metal_silu_ffn(out, gate, up, N)
        bind(FN_SILU_FFN, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT));

        // int gollek_metal_gelu_ffn(out, gate, up, N)
        bind(FN_GELU_FFN, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT));

        bind(FN_SET_MPS_MATVEC_ENABLED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));
        bind(FN_SET_MPS_MATVEC_AUTOTUNE_ENABLED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));
        bind(FN_SET_MPS_MATVEC_MAX_INNER, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));
        bind(FN_SET_MPS_MATVEC_MAX_OUTPUT, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));
        bind(FN_SET_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // int gollek_metal_is_unified_memory()
        bind(FN_UNIFIED_MEM, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // int gollek_metal_device_name(char* buf, int bufSz)
        bind(FN_DEVICE_NAME, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // Basic Math ops
        FunctionDescriptor binaryDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
        FunctionDescriptor unaryDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

        bind(FN_ADD, binaryDesc);
        bind(FN_SUB, binaryDesc);
        bind(FN_MUL, binaryDesc);
        bind(FN_DIV, binaryDesc);

        bind(FN_RELU, unaryDesc);
        bind(FN_SIGMOID, unaryDesc);
        bind(FN_TANH, unaryDesc);
        bind(FN_EXP, unaryDesc);
        bind(FN_LOG, unaryDesc);
        bind(FN_SUM, unaryDesc);
        bind(FN_MEAN, unaryDesc);

        bind(FN_POW, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        bind(FN_TRANSPOSE2D, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> sym = lookup.find(name);
        if (sym.isPresent()) {
            handles.put(name, Linker.nativeLinker().downcallHandle(sym.get(), descriptor));
            LOG.debugf("MetalBinding: bound %s", name);
        } else {
            LOG.debugf("MetalBinding: symbol not found — %s", name);
        }
    }

    /** Reflective invoke helper — wraps Throwable as RuntimeException. */
    private Object invoke(String name, Object... args) {
        MethodHandle mh = handles.get(name);
        if (mh == null)
            throw new IllegalStateException("Unbound: " + name);
        try {
            return mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("MetalBinding." + name + " failed", t);
        }
    }

    // ── Test helper ───────────────────────────────────────────────────────────

    static void reset() {
        instance = null;
    }
}
