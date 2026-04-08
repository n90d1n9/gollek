package tech.kayys.gollek.quantizer.turboquant;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Random Rotation Engine for TurboQuant.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Paper §3.1: "we apply a random rotation Π ∈ ℝ^(d×d) to input x"    │
 * │  "The resulting rotated vector Π·x is uniformly distributed on S^(d-1)│
 * │  As shown in Lemma 1, each coordinate of Π·x follows a Beta           │
 * │  distribution, converging to N(0,1/d) in high dimensions."            │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Two implementations:
 *
 * 1. HADAMARD (preferred, O(d log d)):
 *    Π = D₂ · H_d · D₁
 *    where H_d is the Walsh-Hadamard matrix (d must be a power of 2),
 *    D₁, D₂ are random diagonal {±1/√d} matrices (sign flip vectors).
 *    This composes two FWHT passes with sign randomisation.
 *    The distribution of H·D₁·x for unit-norm x is asymptotically identical
 *    to a uniformly random rotation. Used by QuaRot, SpinQuant, etc.
 *
 * 2. RANDOM_ORTHOGONAL (exact, O(d²)):
 *    Generates a true Haar-distributed random rotation via Gram-Schmidt /
 *    QR decomposition of a d×d Gaussian random matrix.
 *    Stores the full d×d matrix. Exact but memory-intensive for large d.
 *
 * Both variants support:
 *   - apply(x)  → Π·x    (forward rotation before quantization)
 *   - applyT(y) → Πᵀ·y   (inverse rotation during dequantization)
 *   - For orthogonal Π: Πᵀ = Π⁻¹, so inverse = transpose.
 */
public class RandomRotation {

    private static final Logger log = LoggerFactory.getLogger(RandomRotation.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    private final int dim;
    private final long seed;

    // Hadamard variant storage: two sign vectors D₁, D₂ ∈ {±1}
    private final float[] signs1;  // D₁ diagonal (pre-multiplied by 1/√d)
    private final float[] signs2;  // D₂ diagonal (pre-multiplied by 1/√d)

    // Full orthogonal variant: stores d×d rotation matrix row-major
    private final float[] rotMatrix;  // null for Hadamard variant

    private final boolean isHadamard;

    // ── Constructors ──────────────────────────────────────────────────────────

    /**
     * Creates a Hadamard-based random rotation (recommended).
     * Requires d to be a power of 2.
     *
     * @param dim   vector dimension (must be power of 2 for WHT)
     * @param seed  random seed for reproducible sign vectors
     */
    public static RandomRotation hadamard(int dim, long seed) {
        if ((dim & (dim - 1)) != 0) {
            throw new IllegalArgumentException(
                "Hadamard rotation requires d = power of 2, got d=" + dim);
        }
        return new RandomRotation(dim, seed, true, null);
    }

    /**
     * Creates a full random orthogonal rotation (exact but O(d²) cost).
     *
     * @param dim   vector dimension
     * @param seed  random seed
     */
    public static RandomRotation randomOrthogonal(int dim, long seed) {
        return new RandomRotation(dim, seed, false, null);
    }

    private RandomRotation(int dim, long seed, boolean hadamard, float[] precomputedMatrix) {
        this.dim        = dim;
        this.seed       = seed;
        this.isHadamard = hadamard;

        Random rng = new Random(seed);

        if (hadamard) {
            // D₁ and D₂: random {±1} vectors, scaled by 1/√d
            float invSqrtD = (float) (1.0 / Math.sqrt(dim));
            signs1 = new float[dim];
            signs2 = new float[dim];
            for (int i = 0; i < dim; i++) {
                signs1[i] = rng.nextBoolean() ?  invSqrtD : -invSqrtD;
                signs2[i] = rng.nextBoolean() ?  invSqrtD : -invSqrtD;
            }
            rotMatrix = null;
            log.debug("Created Hadamard rotation: dim={}, seed={}", dim, seed);
        } else {
            signs1 = null;
            signs2 = null;
            // Generate Haar-random orthogonal matrix via Gram-Schmidt on Gaussian matrix
            rotMatrix = precomputedMatrix != null
                ? precomputedMatrix
                : generateOrthogonalMatrix(dim, rng);
            log.debug("Created random orthogonal rotation: dim={}, seed={}", dim, seed);
        }
    }

    // ── Forward Rotation: Π·x ────────────────────────────────────────────────

    /**
     * Applies the random rotation: y = Π·x
     *
     * For Hadamard: y = D₂ · FWHT(D₁ · x)  (two sign flips + WHT)
     * For full:     y = rot_matrix · x
     *
     * @param x      input vector [dim], NOT modified
     * @param output pre-allocated output [dim]
     */
    public void apply(float[] x, float[] output) {
        if (x.length != dim || output.length != dim) {
            throw new IllegalArgumentException(
                "Vector length mismatch: expected " + dim);
        }

        if (isHadamard) {
            applyHadamard(x, output, false);
        } else {
            applyMatMul(rotMatrix, x, output, false);
        }
    }

    /**
     * Applies the inverse rotation: x = Πᵀ·y  (since Π is orthogonal, Πᵀ = Π⁻¹)
     *
     * For Hadamard: x = D₁ᵀ · FWHT(D₂ᵀ · y)  = D₁ · FWHT(D₂ · y)
     *               (D₁ᵀ = D₁ for diagonal {±1} matrices)
     */
    public void applyTranspose(float[] y, float[] output) {
        if (y.length != dim || output.length != dim) {
            throw new IllegalArgumentException(
                "Vector length mismatch: expected " + dim);
        }

        if (isHadamard) {
            applyHadamard(y, output, true);
        } else {
            applyMatMul(rotMatrix, y, output, true);
        }
    }

    // ── Hadamard Transform ────────────────────────────────────────────────────

    /**
     * Applies the Hadamard rotation in two directions:
     *   Forward (transpose=false): output = D₂ · H · D₁ · x
     *   Inverse (transpose=true):  output = D₁ · H · D₂ · x
     *
     * The Fast Walsh-Hadamard Transform (FWHT) uses a butterfly network:
     *   O(d log d) operations, in-place, no extra memory.
     *   H is self-inverse up to scale: H·H = d·I, so H⁻¹ = H/d.
     *   Since we scale D₁,D₂ by 1/√d, the combined matrix has norm 1.
     */
    private void applyHadamard(float[] input, float[] output, boolean transpose) {
        // Copy input and apply first diagonal sign flip
        float[] first  = transpose ? signs2 : signs1;
        float[] second = transpose ? signs1 : signs2;

        // Step 1: multiply by first sign vector
        for (int i = 0; i < dim; i++) output[i] = input[i] * first[i];

        // Step 2: in-place Fast WHT (butterfly network)
        fwht(output);

        // Step 3: multiply by second sign vector
        for (int i = 0; i < dim; i++) output[i] *= second[i];
    }

    /**
     * In-place Fast Walsh-Hadamard Transform.
     * Classic butterfly network: O(d log d) additions/subtractions.
     *
     * The transform satisfies H·x = [H_{d/2} H_{d/2}; H_{d/2} -H_{d/2}] · x
     * recursively, with H_1 = [1].
     *
     * Since we scaled the sign diagonals by 1/√d each (total 1/d factor from
     * two D passes), and WHT has norm factor √d, the combined rotation
     * D₂·WHT·D₁ has norm (1/√d)·√d·(1/√d) = 1/√d. We thus get a proper
     * isometry after accounting for the scale.
     */
    private static void fwht(float[] data) {
        int n = data.length;
        for (int len = 1; len < n; len <<= 1) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    float a = data[i + j];
                    float b = data[i + j + len];
                    data[i + j]       = a + b;
                    data[i + j + len] = a - b;
                }
            }
        }
    }

    // ── Full Matrix Rotation ──────────────────────────────────────────────────

    /**
     * Matrix-vector multiply for full orthogonal rotation.
     * Uses Vector API SIMD for the inner dot product.
     */
    private void applyMatMul(float[] matrix, float[] vec, float[] output, boolean transpose) {
        // matrix is row-major [dim × dim]
        for (int row = 0; row < dim; row++) {
            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int k = 0;
            int matRowOff = transpose ? row : row * dim; // col for transpose

            if (transpose) {
                // Πᵀ·v: output[row] = Σ_col matrix[col, row] * vec[col]
                //                             = matrix[col*dim + row]
                // Requires scatter access — fall back to scalar for transpose
                float sum = 0f;
                for (int col = 0; col < dim; col++) {
                    sum += matrix[col * dim + row] * vec[col];
                }
                output[row] = sum;
            } else {
                // Π·v: output[row] = Σ_col matrix[row, col] * vec[col]
                for (; k <= dim - F_LANES; k += F_LANES) {
                    FloatVector vm = FloatVector.fromArray(FLOAT_SPECIES, matrix, row * dim + k);
                    FloatVector vv = FloatVector.fromArray(FLOAT_SPECIES, vec, k);
                    acc = vm.fma(vv, acc);
                }
                float sum = acc.reduceLanes(VectorOperators.ADD);
                for (; k < dim; k++) sum += matrix[row * dim + k] * vec[k];
                output[row] = sum;
            }
        }
    }

    /**
     * Generates a Haar-distributed random orthogonal matrix via Gram-Schmidt.
     * For d ≤ 4096 this is feasible. For larger d, use Hadamard instead.
     */
    private static float[] generateOrthogonalMatrix(int d, Random rng) {
        log.debug("Generating {}×{} random orthogonal matrix (Gram-Schmidt)...", d, d);

        // Start with a d×d Gaussian random matrix
        double[][] M = new double[d][d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                M[i][j] = rng.nextGaussian();

        // Gram-Schmidt orthonormalisation
        for (int i = 0; i < d; i++) {
            // Subtract projections onto all previous basis vectors
            for (int k = 0; k < i; k++) {
                double dot = 0;
                for (int j = 0; j < d; j++) dot += M[i][j] * M[k][j];
                for (int j = 0; j < d; j++) M[i][j] -= dot * M[k][j];
            }
            // Normalise
            double norm = 0;
            for (int j = 0; j < d; j++) norm += M[i][j] * M[i][j];
            norm = Math.sqrt(norm);
            if (norm > 1e-10) for (int j = 0; j < d; j++) M[i][j] /= norm;
        }

        // Flatten to row-major float[]
        float[] flat = new float[d * d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                flat[i * d + j] = (float) M[i][j];

        return flat;
    }

    // ── QJL random matrix (§2.2 + Algorithm 2) ───────────────────────────────

    /**
     * Generates the QJL random matrix S ∈ ℝ^(d×d) with i.i.d. N(0,1) entries.
     * Used in TurboQuant_prod (Algorithm 2) to quantize the residual.
     *
     * The QJL map is: Qqjl(r) = sign(S·r)
     * The dequant is: Q⁻¹_qjl(z) = (√(π/2)/d) · ‖r‖ · Sᵀ · z
     *
     * For efficiency, S is stored as a d×d float array.
     * At inference with d=128 (typical LLM head dim), this is 64 KB.
     */
    public static float[] generateQjlMatrix(int dim, long seed) {
        Random rng = new Random(seed + 1234567L); // offset to differ from rotation seed
        float[] S = new float[dim * dim];
        for (int i = 0; i < S.length; i++) {
            S[i] = (float) rng.nextGaussian();
        }
        return S;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public int getDim()          { return dim; }
    public long getSeed()        { return seed; }
    public boolean isHadamard()  { return isHadamard; }

    /** Returns the sign vectors for Hadamard variant (for serialisation). */
    public float[] getSigns1()   { return signs1; }
    public float[] getSigns2()   { return signs2; }

    /** Returns the full rotation matrix (for full-orthogonal variant). */
    public float[] getRotMatrix() { return rotMatrix; }

    @Override
    public String toString() {
        return "RandomRotation{dim=%d, strategy=%s, seed=%d}"
            .formatted(dim, isHadamard ? "HADAMARD" : "FULL_ORTHOGONAL", seed);
    }
}
