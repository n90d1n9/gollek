package tech.kayys.gollek.quantizer.quip;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * QuIP# quantizer — implements the algorithm from:
 * <em>"QuIP#: Even Better LLM Quantization with Hadamard Incoherence and
 * Lattice Codebooks"</em> (Tseng et al., 2024, arxiv 2406.11235).
 *
 * <h3>Algorithm (§2–3 of the paper)</h3>
 * <ol>
 *   <li><b>Incoherence processing</b> — apply random Hadamard transforms
 *       {@code U} (rows) and {@code V} (cols) to the weight matrix:
 *       {@code W̃ = U W V^T}. This makes the weight distribution more
 *       uniform, reducing the quantization error from O(‖W‖_∞) to
 *       O(‖W‖_F / √n).</li>
 *   <li><b>E8 vector quantization</b> — partition the transformed weights
 *       into non-overlapping 8-dimensional blocks and find the nearest
 *       E8 lattice codeword for each block. Store the codeword index
 *       (1 byte) and a per-block scale factor (float32).</li>
 *   <li><b>Dequantization</b> — reconstruct W̃ from codes + scales, then
 *       apply the inverse transforms: {@code W ≈ U^T W̃ V}.</li>
 * </ol>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * QuipQuantizer q = new QuipQuantizer(QuipConfig.quip2bit());
 *
 * // Quantize a weight matrix [rows × cols]
 * QuipTensor qt = q.quantize("layer.weight", weights, rows, cols);
 *
 * // Dequantize back to float[]
 * float[] approx = q.dequantize(qt);
 * }</pre>
 */
public class QuipQuantizer {

    private static final Logger log = LoggerFactory.getLogger(QuipQuantizer.class);

    private final QuipConfig config;

    public QuipQuantizer(QuipConfig config) {
        this.config = config;
    }

    // ── Quantize ──────────────────────────────────────────────────────────────

    /**
     * Quantize a weight matrix using QuIP# incoherence + E8 VQ.
     *
     * @param name  tensor name (for logging)
     * @param W     row-major float[] of shape [rows × cols]
     * @param rows  number of rows
     * @param cols  number of columns
     * @return compressed {@link QuipTensor}
     */
    public QuipTensor quantize(String name, float[] W, int rows, int cols) {
        if (W.length != rows * cols)
            throw new IllegalArgumentException("W.length != rows*cols");

        log.debug("QuIP# quantizing {} [{}×{}]", name, rows, cols);

        // Work on a copy — do not mutate the original
        float[] Wt = Arrays.copyOf(W, W.length);

        // ── Step 1: Incoherence processing ────────────────────────────────────
        // Apply U to rows (left transform) and V to cols (right transform).
        // Pad dimensions to next power of 2 if needed.

        int rowPad = nextPow2(cols); // V acts on cols
        int colPad = nextPow2(rows); // U acts on rows

        RandomHadamard V = new RandomHadamard(rowPad, config.hadamardSeedV());
        RandomHadamard U = new RandomHadamard(colPad, config.hadamardSeedU());

        // Apply V^T to each row: W̃[r,:] = V * W[r,:]
        if (rowPad == cols) {
            V.applyRows(Wt, rows);
        } else {
            Wt = padAndApplyRows(Wt, rows, cols, V, rowPad);
        }

        // Apply U to each col: W̃[:,c] = U * W̃[:,c]
        if (colPad == rows) {
            U.applyCols(Wt, cols);
        } else {
            Wt = padAndApplyCols(Wt, rows, cols, U, colPad);
        }

        // ── Step 2: E8 vector quantization ────────────────────────────────────
        int n = rows * cols;
        int numBlocks = n / E8Codebook.DIM;
        int remainder = n % E8Codebook.DIM;

        byte[]  codes  = new byte[numBlocks + (remainder > 0 ? 1 : 0)];
        float[] scales = new float[codes.length];

        float[] block = new float[E8Codebook.DIM];

        for (int b = 0; b < numBlocks; b++) {
            int off = b * E8Codebook.DIM;
            System.arraycopy(Wt, off, block, 0, E8Codebook.DIM);

            // Per-block scale: normalise to unit norm before codebook lookup
            float norm = blockNorm(block);
            scales[b] = norm;
            if (norm > 1e-8f) {
                float inv = 1f / norm;
                for (int d = 0; d < E8Codebook.DIM; d++) block[d] *= inv;
            }

            codes[b] = (byte) E8Codebook.nearestIndex(block);
        }

        // Handle remainder with scalar fallback
        if (remainder > 0) {
            int b = numBlocks;
            int off = b * E8Codebook.DIM;
            Arrays.fill(block, 0f);
            System.arraycopy(Wt, off, block, 0, remainder);
            float norm = blockNorm(block);
            scales[b] = norm;
            if (norm > 1e-8f) {
                float inv = 1f / norm;
                for (int d = 0; d < remainder; d++) block[d] *= inv;
            }
            codes[b] = (byte) E8Codebook.nearestIndex(block);
        }

        log.debug("QuIP# {} → {} blocks, {:.2f}x compression",
                name, codes.length, (double)(rows * cols * 4) / (codes.length + scales.length * 4));

        return new QuipTensor(name, rows, cols, codes, scales,
                config.hadamardSeedU(), config.hadamardSeedV());
    }

    // ── Dequantize ────────────────────────────────────────────────────────────

    /**
     * Reconstruct an approximate weight matrix from a {@link QuipTensor}.
     *
     * @param qt quantized tensor
     * @return float[] of shape [rows × cols]
     */
    public float[] dequantize(QuipTensor qt) {
        int rows = qt.rows(), cols = qt.cols();
        int n = rows * cols;
        float[] Wt = new float[n];

        // Reconstruct from E8 codewords + scales
        int numBlocks = n / E8Codebook.DIM;
        int remainder = n % E8Codebook.DIM;

        for (int b = 0; b < numBlocks; b++) {
            float[] cw = E8Codebook.codeword(qt.codes()[b] & 0xFF);
            float scale = qt.scales()[b];
            int off = b * E8Codebook.DIM;
            for (int d = 0; d < E8Codebook.DIM; d++) Wt[off + d] = cw[d] * scale;
        }

        if (remainder > 0) {
            int b = numBlocks;
            float[] cw = E8Codebook.codeword(qt.codes()[b] & 0xFF);
            float scale = qt.scales()[b];
            int off = b * E8Codebook.DIM;
            for (int d = 0; d < remainder; d++) Wt[off + d] = cw[d] * scale;
        }

        // ── Inverse incoherence processing ────────────────────────────────────
        // W ≈ U^T W̃ V^T  (both H and D are self-inverse up to scale)

        int rowPad = nextPow2(cols);
        int colPad = nextPow2(rows);

        RandomHadamard V = new RandomHadamard(rowPad, qt.seedV());
        RandomHadamard U = new RandomHadamard(colPad, qt.seedU());

        // Inverse U on cols
        if (colPad == rows) {
            applyInverseCols(Wt, rows, cols, U);
        }

        // Inverse V on rows
        if (rowPad == cols) {
            applyInverseRows(Wt, rows, cols, V);
        }

        return Wt;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float blockNorm(float[] block) {
        float s = 0f;
        for (float v : block) s += v * v;
        return (float) Math.sqrt(s);
    }

    private static int nextPow2(int n) {
        if (n <= 1) return 1;
        return Integer.highestOneBit(n - 1) << 1;
    }

    /** Pad rows to rowPad, apply V, then trim back. */
    private static float[] padAndApplyRows(float[] W, int rows, int cols,
                                            RandomHadamard V, int rowPad) {
        float[] padded = new float[rows * rowPad];
        for (int r = 0; r < rows; r++)
            System.arraycopy(W, r * cols, padded, r * rowPad, cols);
        V.applyRows(padded, rows);
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            System.arraycopy(padded, r * rowPad, out, r * cols, cols);
        return out;
    }

    /** Pad cols to colPad, apply U, then trim back. */
    private static float[] padAndApplyCols(float[] W, int rows, int cols,
                                            RandomHadamard U, int colPad) {
        float[] padded = new float[colPad * cols];
        for (int r = 0; r < rows; r++)
            System.arraycopy(W, r * cols, padded, r * cols, cols);
        U.applyCols(padded, cols);
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            System.arraycopy(padded, r * cols, out, r * cols, cols);
        return out;
    }

    private static void applyInverseRows(float[] W, int rows, int cols, RandomHadamard V) {
        float[] row = new float[cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(W, r * cols, row, 0, cols);
            V.applyInverse(row);
            System.arraycopy(row, 0, W, r * cols, cols);
        }
    }

    private static void applyInverseCols(float[] W, int rows, int cols, RandomHadamard U) {
        float[] col = new float[rows];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) col[r] = W[r * cols + c];
            U.applyInverse(col);
            for (int r = 0; r < rows; r++) W[r * cols + c] = col[r];
        }
    }
}
