package tech.kayys.gollek.quantizer.turboquant;

import tech.kayys.gollek.quantizer.turboquant.RandomRotation;
import tech.kayys.gollek.quantizer.turboquant.LloydMaxCodebook;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantConfig;
import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TurboQuant Quantization Engine — JDK 25 Vector API.
 *
 * Implements both Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
 * from the paper "TurboQuant: Online Vector Quantization with Near-optimal
 * Distortion Rate" (Zandieh et al., arXiv:2504.19874v1).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Algorithm 1: TurboQuant_mse                                           │
 * │                                                                        │
 * │  Setup:                                                                │
 * │    1. Generate random rotation Π (Hadamard or full orthogonal)        │
 * │    2. Precompute Lloyd-Max codebook C* for N(0, 1/d) source           │
 * │                                                                        │
 * │  Quant(x):                                                            │
 * │    y ← Π · x                          [rotate to spherical coords]   │
 * │    idx[j] ← argmin_k |y[j] − c_k|    [nearest centroid per coord]   │
 * │    output: idx ∈ {0,...,2^b-1}^d                                      │
 * │                                                                        │
 * │  DeQuant(idx):                                                         │
 * │    ỹ[j] ← c_{idx[j]}                  [centroid lookup]              │
 * │    x̃ ← Πᵀ · ỹ                         [rotate back]                 │
 * │    output: x̃ ∈ ℝ^d                                                   │
 * │                                                                        │
 * │  Algorithm 2: TurboQuant_prod                                          │
 * │                                                                        │
 * │  Setup:                                                                │
 * │    (same as Qmse with bits b-1)                                        │
 * │    S ∈ ℝ^(d×d) with S_{i,j} ~ N(0,1)  [QJL random matrix]           │
 * │                                                                        │
 * │  Quant(x):                                                            │
 * │    idx ← Quant_mse(x)                 [stage 1: MSE quant at b-1]   │
 * │    r ← x − DeQuant_mse(idx)           [residual vector]              │
 * │    γ ← ‖r‖₂                           [store residual norm]          │
 * │    qjl ← sign(S · r)                  [stage 2: 1-bit QJL]          │
 * │    output: (idx, qjl, γ)                                              │
 * │                                                                        │
 * │  DeQuant(idx, qjl, γ):                                                │
 * │    x̃_mse ← DeQuant_mse(idx)                                          │
 * │    x̃_qjl ← (√(π/2)/d) · γ · Sᵀ · qjl                               │
 * │    output: x̃_mse + x̃_qjl                                            │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Vector API usage:
 *   - Nearest-centroid search: binary search (scalar, cache-efficient)
 *   - Centroid lookup + residual: FloatVector.fromArray, sub, fma
 *   - QJL: sign of matrix-vector product via FloatVector comparisons
 *   - L2 norm: FloatVector.mul + reduceLanes(ADD) + Math.sqrt
 *   - Inner product estimation: FloatVector.fma for Sᵀ·z
 */
public class TurboQuantEngine {

    private static final Logger log = LoggerFactory.getLogger(TurboQuantEngine.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    private final TurboQuantConfig config;
    private final int              dim;

    // MSE stage components
    private final RandomRotation   rotation;
    private final LloydMaxCodebook codebook;   // for MSE stage (bits or bits-1)

    // QJL components (only for INNER_PRODUCT variant)
    private final float[]          qjlMatrix;  // S ∈ ℝ^(d×d), null for MSE variant

    // Scratch buffers (thread-local to avoid allocation on hot path)
    // Note: Initialized lazily to avoid referencing config before constructor
    private ThreadLocal<float[]> scratchRotated;
    private ThreadLocal<float[]> scratchDeQuant;
    private ThreadLocal<float[]> scratchResidual;
    private ThreadLocal<int[]>   scratchIndices;

    // ── Constructor ───────────────────────────────────────────────────────────

    public TurboQuantEngine(TurboQuantConfig config) {
        this.config = config;
        this.dim    = config.dimension();

        // Build rotation
        this.rotation = switch (config.rotation()) {
            case HADAMARD         -> RandomRotation.hadamard(dim, config.seed());
            case RANDOM_ORTHOGONAL -> RandomRotation.randomOrthogonal(dim, config.seed());
            case RANDOM_SVD       -> RandomRotation.hadamard(dim, config.seed()); // fallback
        };

        // Build Lloyd-Max codebook for the MSE stage
        // For Qprod: the MSE stage uses (bits-1) bits (§3.2 of paper)
        this.codebook = new LloydMaxCodebook(config.mseStageBits(), dim);

        // Build QJL matrix only for inner-product variant
        this.qjlMatrix = config.isInnerProductVariant()
            ? RandomRotation.generateQjlMatrix(dim, config.seed())
            : null;

        // Initialize scratch buffers (lazy initialization to avoid config reference before constructor)
        this.scratchRotated  = ThreadLocal.withInitial(() -> new float[config.dimension()]);
        this.scratchDeQuant  = ThreadLocal.withInitial(() -> new float[config.dimension()]);
        this.scratchResidual = ThreadLocal.withInitial(() -> new float[config.dimension()]);
        this.scratchIndices  = ThreadLocal.withInitial(() -> new int[config.dimension()]);

        log.info("TurboQuantEngine: config={}, codebook={}", config, codebook);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Algorithm 1: TurboQuant_mse — Quantise/Dequantise
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Quantizes a single vector using TurboQuant_mse (Algorithm 1).
     *
     * @param x       input vector [dim], unit-norm assumed (‖x‖=1)
     * @param indices output quantization indices [dim], pre-allocated
     */
    public void quantizeMse(float[] x, int[] indices) {
        float[] rotated = scratchRotated.get();
        rotation.apply(x, rotated);
        codebook.quantize(rotated, indices);
    }

    /**
     * Dequantizes indices back to a reconstructed vector (TurboQuant_mse).
     *
     * @param indices input quantization indices [dim]
     * @param output  reconstructed vector [dim], pre-allocated
     */
    public void dequantizeMse(int[] indices, float[] output) {
        float[] rotated = scratchRotated.get();
        codebook.dequantize(indices, rotated);
        rotation.applyTranspose(rotated, output);
    }

    /**
     * Batch quantize: applies quantizeMse to each row of a matrix.
     *
     * @param matrix  input [numVectors, dim], row-major
     * @param indices output [numVectors, dim], row-major
     * @param numVecs number of vectors
     */
    public void quantizeMseBatch(float[] matrix, int[] indices, int numVecs) {
        float[] xBuf  = new float[dim];
        int[]   iBuf  = new int[dim];
        for (int v = 0; v < numVecs; v++) {
            System.arraycopy(matrix, v * dim, xBuf, 0, dim);
            quantizeMse(xBuf, iBuf);
            System.arraycopy(iBuf, 0, indices, v * dim, dim);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Algorithm 2: TurboQuant_prod — Quantise/Dequantise
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Result of TurboQuant_prod quantization (Algorithm 2, lines 5-8).
     *
     * @param mseIndices  quantization indices from the MSE stage    [dim]
     * @param qjlSigns    1-bit QJL signs ∈ {-1, +1} for the residual [dim]
     * @param residualNorm ‖r‖₂, the L2 norm of the MSE residual
     */
    public record QuantProdResult(int[] mseIndices, byte[] qjlSigns, float residualNorm) {
        /** Total bits used: b*d bits for mseIndices + 1*d bits for qjlSigns (≈ b*d total) */
        public int totalBits(int bits) { return mseIndices.length * bits + qjlSigns.length; }
    }

    /**
     * Quantizes using TurboQuant_prod (Algorithm 2).
     * Unbiased inner-product estimator: E[⟨y, x̃⟩] = ⟨y, x⟩.
     *
     * @param x  input vector [dim], unit-norm (‖x‖=1)
     * @return   QuantProdResult with (idx, qjl, ‖r‖)
     */
    public QuantProdResult quantizeProd(float[] x) {
        if (!config.isInnerProductVariant()) {
            throw new IllegalStateException(
                "quantizeProd requires INNER_PRODUCT variant; current: " + config.variant());
        }

        // Stage 1: MSE quantize at (bits-1) bits
        int[]   mseIndices = new int[dim];
        float[] xMse       = scratchDeQuant.get();
        quantizeMse(x, mseIndices);
        dequantizeMse(mseIndices, xMse);

        // Compute residual r = x − x̃_mse  (line 6 of Algorithm 2)
        float[] r = scratchResidual.get();
        float   normR = computeResidual(x, xMse, r);

        // Stage 2: QJL on residual  qjl = sign(S · r)  (line 7)
        byte[] qjlSigns = applyQjl(r);

        return new QuantProdResult(mseIndices, qjlSigns, normR);
    }

    /**
     * Dequantizes TurboQuant_prod result (Algorithm 2, lines 9-12).
     *
     * x̃ = x̃_mse + (√(π/2)/d) · γ · Sᵀ · qjl
     *
     * @param result  the QuantProdResult from quantizeProd
     * @param output  reconstructed vector [dim], pre-allocated
     */
    public void dequantizeProd(QuantProdResult result, float[] output) {
        // x̃_mse (line 10)
        dequantizeMse(result.mseIndices(), output);

        // Compute QJL dequantization: x̃_qjl = (√(π/2)/d) · γ · Sᵀ · qjl  (line 11)
        float[] xQjl = scratchResidual.get();
        dequantizeQjl(result.qjlSigns(), result.residualNorm(), xQjl);

        // x̃ = x̃_mse + x̃_qjl  (line 12)
        vectorAdd(output, xQjl);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Inner Product Estimation
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Estimates ⟨y, x⟩ from a QuantProdResult without full dequantization.
     *
     * ⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + (√(π/2)/d) · γ · Σ_i qjl_i · (Sᵀ_i · y)
     *
     * This is the critical hot path for KV cache attention computation.
     * All operations are SIMD-vectorized.
     *
     * @param y       query vector [dim] (key or value)
     * @param xMse    dequantized MSE component of x [dim]
     * @param result  QuantProdResult for x
     * @return        estimated inner product ⟨y, x⟩
     */
    public float estimateInnerProduct(float[] y, float[] xMse, QuantProdResult result) {
        // Term 1: ⟨y, x̃_mse⟩
        float term1 = dotProduct(y, xMse);

        // Term 2: (√(π/2)/d) · γ · Σ_i sign_i · (Sᵀ·y)_i
        // = (√(π/2)/d) · γ · Σ_i sign_i · Σ_j S[j,i] · y[j]
        float coeff = (float) (Math.sqrt(Math.PI / 2.0) / dim) * result.residualNorm();
        float term2 = qjlInnerProduct(y, result.qjlSigns(), coeff);

        return term1 + term2;
    }

    /**
     * Unbiased inner product estimate directly from raw quantized data.
     * Combines MSE dequantization + QJL estimation in one pass.
     */
    public float estimateInnerProductFull(float[] y, QuantProdResult result) {
        float[] xMse = scratchDeQuant.get();
        dequantizeMse(result.mseIndices(), xMse);
        return estimateInnerProduct(y, xMse, result);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QJL Implementation (§2.2, Definition 1)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Applies the QJL map: qjl = sign(S · r)
     *
     * S ∈ ℝ^(d×d) with i.i.d. N(0,1) entries.
     * sign is applied element-wise.
     *
     * @param r  residual vector [dim]
     * @return   sign vector ∈ {-1, +1}^d, packed as bytes (1 byte per element)
     */
    private byte[] applyQjl(float[] r) {
        byte[] signs = new byte[dim];

        for (int i = 0; i < dim; i++) {
            // Compute (S · r)[i] = Σ_j S[i,j] · r[j]
            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int j = 0;
            for (; j <= dim - F_LANES; j += F_LANES) {
                FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, qjlMatrix, i * dim + j);
                FloatVector vr = FloatVector.fromArray(FLOAT_SPECIES, r, j);
                acc = vs.fma(vr, acc);
            }
            float dot = acc.reduceLanes(VectorOperators.ADD);
            for (; j < dim; j++) dot += qjlMatrix[i * dim + j] * r[j];

            signs[i] = dot >= 0 ? (byte) 1 : (byte) -1;
        }
        return signs;
    }

    /**
     * Dequantizes QJL: x̃_qjl = (√(π/2)/d) · γ · Sᵀ · qjl
     *
     * Per Definition 1 of the paper: Q⁻¹_qjl(z) = (√(π/2)/d) · Sᵀ · z
     * where z ∈ {-1,+1}^d. We additionally scale by γ = ‖r‖₂.
     *
     * @param qjlSigns  sign vector [dim] ∈ {-1, +1} as bytes
     * @param gamma     residual norm ‖r‖₂
     * @param output    dequantized residual estimate [dim]
     */
    private void dequantizeQjl(byte[] qjlSigns, float gamma, float[] output) {
        float scale = (float) (Math.sqrt(Math.PI / 2.0) / dim) * gamma;

        for (int j = 0; j < dim; j++) {
            // (Sᵀ · qjl)[j] = Σ_i S[i,j] · qjl[i]
            float sum = 0f;
            int i = 0;
            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

            for (; i <= dim - F_LANES; i += F_LANES) {
                // S is row-major, so S[i,j] = qjlMatrix[i*dim + j]
                // We need the j-th column of S — stride access (not contiguous)
                // For performance we use scalar here and rely on JIT auto-vectorization
                // A transposed copy of S (S_T) would make this fully SIMD-friendly
                // (stored optionally as qjlMatrixT)
                for (int lane = 0; lane < F_LANES && i + lane < dim; lane++) {
                    sum += qjlMatrix[(i + lane) * dim + j] * qjlSigns[i + lane];
                }
            }
            for (; i < dim; i++) {
                sum += qjlMatrix[i * dim + j] * qjlSigns[i];
            }

            output[j] = scale * sum;
        }
    }

    /**
     * Computes the QJL inner product contribution: (√(π/2)/d) · γ · (Sᵀ·qjl)·y
     * Used in estimateInnerProduct for the attention kernel.
     */
    private float qjlInnerProduct(float[] y, byte[] qjlSigns, float coeff) {
        // = coeff · Σ_j y[j] · (Sᵀ·qjl)[j]
        // = coeff · Σ_j y[j] · Σ_i S[i,j]·qjl[i]
        // = coeff · Σ_i qjl[i] · (S[i,:]·y)   ← reorder sum
        float total = 0f;
        for (int i = 0; i < dim; i++) {
            // Compute S[i,:]·y = Σ_j S[i,j]·y[j]  ← row access (contiguous)
            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int j = 0;
            for (; j <= dim - F_LANES; j += F_LANES) {
                FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, qjlMatrix, i * dim + j);
                FloatVector vy = FloatVector.fromArray(FLOAT_SPECIES, y, j);
                acc = vs.fma(vy, acc);
            }
            float dot = acc.reduceLanes(VectorOperators.ADD);
            for (; j < dim; j++) dot += qjlMatrix[i * dim + j] * y[j];

            total += qjlSigns[i] * dot;
        }
        return coeff * total;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MSE Distortion Computation
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Computes the actual per-vector MSE distortion ‖x − x̃‖²₂.
     * Used to validate against the theoretical bound (Theorem 1).
     */
    public float computeMseDistortion(float[] x, int[] mseIndices) {
        float[] xTilde = scratchDeQuant.get();
        dequantizeMse(mseIndices, xTilde);

        FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
        int i = 0;
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, x,      i);
            FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, xTilde, i);
            FloatVector diff = vx.sub(vq);
            acc = diff.fma(diff, acc);
        }
        float mse = acc.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            float d = x[i] - xTilde[i];
            mse += d * d;
        }
        return mse;
    }

    /**
     * Computes the inner product distortion ⟨y,x⟩ − ⟨y,x̃⟩.
     * Used to validate Theorem 2.
     */
    public float computeInnerProductError(float[] x, float[] y, QuantProdResult result) {
        float trueIP  = dotProduct(y, x);
        float estIP   = estimateInnerProductFull(y, result);
        return trueIP - estIP;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vector Utilities (SIMD)
    // ═══════════════════════════════════════════════════════════════════════

    /** Computes residual r = x − xMse and returns ‖r‖₂ */
    private float computeResidual(float[] x, float[] xMse, float[] r) {
        FloatVector normAcc = FloatVector.zero(FLOAT_SPECIES);
        int i = 0;
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, x,    i);
            FloatVector vm = FloatVector.fromArray(FLOAT_SPECIES, xMse, i);
            FloatVector vr = vx.sub(vm);
            vr.intoArray(r, i);
            normAcc = vr.fma(vr, normAcc);
        }
        float normSq = normAcc.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            float ri = x[i] - xMse[i];
            r[i]    = ri;
            normSq += ri * ri;
        }
        return (float) Math.sqrt(normSq);
    }

    /** SIMD L2 norm ‖x‖₂ */
    public float norm(float[] x) {
        FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
        int i = 0;
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector v = FloatVector.fromArray(FLOAT_SPECIES, x, i);
            acc = v.fma(v, acc);
        }
        float sq = acc.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) sq += x[i] * x[i];
        return (float) Math.sqrt(sq);
    }

    /** SIMD dot product ⟨a, b⟩ */
    public float dotProduct(float[] a, float[] b) {
        FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
        int i = 0;
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
            acc = va.fma(vb, acc);
        }
        float dot = acc.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) dot += a[i] * b[i];
        return dot;
    }

    /** In-place vector add: a += b */
    private void vectorAdd(float[] a, float[] b) {
        int i = 0;
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
            va.add(vb).intoArray(a, i);
        }
        for (; i < dim; i++) a[i] += b[i];
    }

    /** Normalises x in-place: x ← x / ‖x‖₂ */
    public void normalize(float[] x) {
        float n = norm(x);
        if (n < 1e-10f) return;
        float invN = 1f / n;
        int i = 0;
        FloatVector vInvN = FloatVector.broadcast(FLOAT_SPECIES, invN);
        for (; i <= dim - F_LANES; i += F_LANES) {
            FloatVector.fromArray(FLOAT_SPECIES, x, i).mul(vInvN).intoArray(x, i);
        }
        for (; i < dim; i++) x[i] *= invN;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public TurboQuantConfig getConfig()        { return config; }
    public LloydMaxCodebook getCodebook()      { return codebook; }
    public RandomRotation   getRotation()      { return rotation; }
    public int              getDim()           { return dim; }

    public static void printCapabilities() {
        System.out.println("=== TurboQuant Engine ===");
        System.out.printf("Paper: arXiv:2504.19874 (Zandieh et al., 2025)%n");
        System.out.printf("MSE bound:  Dmse ≤ √(3π)/2 · 4^(-b) ≈ 2.72 · 4^(-b)%n");
        System.out.printf("IP  bound:  Dprod ≤ √(3π)/2 · ‖y‖²/d · 4^(-b)%n");
        System.out.printf("Lower bnd:  Dmse ≥ 4^(-b)  (Theorem 3, info-theoretic)%n");
        System.out.printf("SIMD:       %s (%d float lanes)%n",
            FLOAT_SPECIES.toString(), FLOAT_SPECIES.length());
    }
}
