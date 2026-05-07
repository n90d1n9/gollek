package tech.kayys.gollek.quantizer.quip;

import java.util.random.RandomGenerator;

/**
 * Randomized Hadamard Transform (RHT) for incoherence processing.
 *
 * <p>QuIP# §2 — Incoherence Processing: before quantizing weight matrix W,
 * apply random orthogonal transforms:
 * <pre>
 *   W̃ = U W V^T
 * </pre>
 * where U, V are random Hadamard matrices (scaled by ±1 diagonal).
 * This spreads quantization error uniformly across all weights, reducing
 * the worst-case error from O(‖W‖_∞) to O(‖W‖_F / √n).
 *
 * <p>The Hadamard transform of size n (power of 2) runs in O(n log n) via
 * the Fast Walsh-Hadamard Transform (FWHT).
 */
public final class RandomHadamard {

    private final float[] diag; // random ±1 diagonal
    private final int n;

    /**
     * Create a random Hadamard transform of size {@code n} (must be power of 2).
     *
     * @param n    dimension (power of 2)
     * @param seed random seed for reproducibility
     */
    public RandomHadamard(int n, long seed) {
        if (Integer.bitCount(n) != 1) throw new IllegalArgumentException("n must be power of 2, got " + n);
        this.n    = n;
        this.diag = randomDiag(n, seed);
    }

    /**
     * Apply the transform in-place: {@code x ← H·D·x / √n}.
     * D is the random ±1 diagonal, H is the Walsh-Hadamard matrix.
     */
    public void apply(float[] x) {
        if (x.length != n) throw new IllegalArgumentException("length mismatch");
        // Step 1: apply diagonal
        for (int i = 0; i < n; i++) x[i] *= diag[i];
        // Step 2: FWHT
        fwht(x);
        // Step 3: normalise
        float scale = 1f / (float) Math.sqrt(n);
        for (int i = 0; i < n; i++) x[i] *= scale;
    }

    /**
     * Apply the inverse transform in-place: {@code x ← D·H^T·x · √n}.
     * Since H is symmetric and D² = I, inverse = D·H / √n applied again.
     */
    public void applyInverse(float[] x) {
        // H is its own inverse (up to scale), so H^{-1} = H/n
        // Full inverse: x = D * H * (H * D * x / √n) * √n / n = D * H * y
        float scale = (float) Math.sqrt(n);
        for (int i = 0; i < n; i++) x[i] *= scale;
        fwht(x);
        float invN = 1f / n;
        for (int i = 0; i < n; i++) x[i] = x[i] * invN * diag[i];
    }

    /**
     * Apply the transform to each row of a matrix stored row-major.
     *
     * @param W    row-major matrix of shape [rows × n]
     * @param rows number of rows
     */
    public void applyRows(float[] W, int rows) {
        float[] row = new float[n];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(W, r * n, row, 0, n);
            apply(row);
            System.arraycopy(row, 0, W, r * n, n);
        }
    }

    /**
     * Apply the transform to each column of a matrix stored row-major.
     *
     * @param W    row-major matrix of shape [n × cols]
     * @param cols number of columns
     */
    public void applyCols(float[] W, int cols) {
        float[] col = new float[n];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < n; r++) col[r] = W[r * cols + c];
            apply(col);
            for (int r = 0; r < n; r++) W[r * cols + c] = col[r];
        }
    }

    // ── Fast Walsh-Hadamard Transform ─────────────────────────────────────────

    static void fwht(float[] x) {
        int n = x.length;
        for (int len = 1; len < n; len <<= 1) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    float a = x[i + j];
                    float b = x[i + j + len];
                    x[i + j]       = a + b;
                    x[i + j + len] = a - b;
                }
            }
        }
    }

    private static float[] randomDiag(int n, long seed) {
        RandomGenerator rng = RandomGenerator.of("Xoshiro256PlusPlus");
        // seed via advance — use a simple LCG to set initial state
        float[] d = new float[n];
        java.util.Random r = new java.util.Random(seed);
        for (int i = 0; i < n; i++) d[i] = r.nextBoolean() ? 1f : -1f;
        return d;
    }
}
