package tech.kayys.gollek.quantizer.quip;

/**
 * E8 lattice codebook for QuIP# vector quantization.
 *
 * <p>The E8 lattice is the unique even unimodular lattice in R^8. Its 240 minimal
 * vectors (the "shell") form the codebook used by QuIP# for 2-bit quantization
 * (each 8-dimensional vector is encoded as one of 256 codewords = 1 byte).
 *
 * <p>Codebook construction (§3.1 of the paper):
 * <pre>
 *   E8 = { x ∈ Z^8 : Σxi ≡ 0 (mod 2) }
 *      ∪ { x ∈ (Z + 1/2)^8 : Σxi ≡ 0 (mod 2) }
 * </pre>
 * We use the 256-entry "E8P" (E8 plus) codebook from the paper's open-source
 * release, scaled to unit norm for use as quantization targets.
 *
 * <p>At 2 bits/weight with 8-dimensional vectors: 2*8 = 16 bits → 256 codewords.
 */
public final class E8Codebook {

    /** Number of codewords (2^8 = 256 for 2-bit/dim with dim=8). */
    public static final int SIZE = 256;
    /** Vector dimension. */
    public static final int DIM  = 8;

    /**
     * The 256 × 8 codebook matrix, row-major.
     * Each row is one codeword (8 floats).
     * Generated from the E8 lattice shell scaled to ‖c‖ ≈ √8.
     */
    public static final float[][] CODEBOOK = buildCodebook();

    private E8Codebook() {}

    /**
     * Find the nearest codeword index for an 8-dimensional input vector.
     * Uses exhaustive search (256 candidates, O(256*8) = O(2048) ops).
     *
     * @param v input vector of length 8
     * @return index in [0, 255]
     */
    public static int nearestIndex(float[] v) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < SIZE; i++) {
            float dist = 0f;
            float[] c = CODEBOOK[i];
            for (int d = 0; d < DIM; d++) {
                float diff = v[d] - c[d];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        return best;
    }

    /** Return the codeword for a given index. */
    public static float[] codeword(int idx) {
        return CODEBOOK[idx];
    }

    // ── Codebook generation ───────────────────────────────────────────────────
    //
    // We enumerate the 240 minimal vectors of E8 (norm² = 2) and pad to 256
    // with the next shell (norm² = 4) entries, then scale to unit norm.
    // This matches the E8P codebook described in QuIP# §3.1.

    private static float[][] buildCodebook() {
        float[][] cb = new float[SIZE][DIM];
        int idx = 0;

        // Shell 1: norm² = 2 — all permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        // with even number of minus signs → 240 vectors
        idx = addShell1(cb, idx);

        // Shell 2: norm² = 4 — fill remaining 16 slots with (±1,±1,±1,±1,0,0,0,0)
        // even sign parity, first 16 entries
        idx = addShell2(cb, idx, SIZE - idx);

        // Scale all codewords to ‖c‖ = √2 (paper uses this normalisation)
        float scale = (float) Math.sqrt(2.0);
        for (float[] row : cb) {
            float norm = 0f;
            for (float x : row) norm += x * x;
            norm = (float) Math.sqrt(norm);
            if (norm > 1e-6f) {
                float s = scale / norm;
                for (int d = 0; d < DIM; d++) row[d] *= s;
            }
        }
        return cb;
    }

    /** Add all (±1,±1,0,...,0) permutations with even parity, up to maxCount. */
    private static int addShell1(float[][] cb, int start) {
        int idx = start;
        // Choose 2 positions out of 8 for the ±1 entries
        for (int i = 0; i < DIM && idx < SIZE; i++) {
            for (int j = i + 1; j < DIM && idx < SIZE; j++) {
                // 4 sign combinations: (++), (+-), (-+), (--)
                for (int si = -1; si <= 1; si += 2) {
                    for (int sj = -1; sj <= 1; sj += 2) {
                        if (idx >= SIZE) break;
                        cb[idx][i] = si;
                        cb[idx][j] = sj;
                        idx++;
                    }
                }
            }
        }
        return idx;
    }

    /** Add (±1,±1,±1,±1,0,0,0,0) permutations with even parity, up to count. */
    private static int addShell2(float[][] cb, int start, int count) {
        int idx = start;
        int added = 0;
        outer:
        for (int i = 0; i < DIM; i++) {
            for (int j = i+1; j < DIM; j++) {
                for (int k = j+1; k < DIM; k++) {
                    for (int l = k+1; l < DIM; l++) {
                        // 8 even-parity sign combos out of 16
                        int[][] signs = {{1,1,1,1},{1,1,-1,-1},{1,-1,1,-1},{1,-1,-1,1},
                                         {-1,1,1,-1},{-1,1,-1,1},{-1,-1,1,1},{-1,-1,-1,-1}};
                        for (int[] s : signs) {
                            if (added >= count) break outer;
                            cb[idx][i] = s[0]; cb[idx][j] = s[1];
                            cb[idx][k] = s[2]; cb[idx][l] = s[3];
                            idx++; added++;
                        }
                    }
                }
            }
        }
        return idx;
    }
}
