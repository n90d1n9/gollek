package tech.kayys.gollek.quantizer.turboquant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lloyd-Max Optimal Scalar Quantizer Codebook for TurboQuant.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Paper §3.1: MSE-Optimal Scalar Quantization                          │
 * │                                                                        │
 * │  After random rotation, each coordinate yⱼ follows a Beta distribution│
 * │  (Lemma 1): yⱼ ~ fX(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)│
 * │  For large d this converges to N(0, 1/d).                             │
 * │                                                                        │
 * │  The optimal quantization centroids c₁,...,c_{2^b} solve the          │
 * │  continuous k-means problem (Eq. 4 of the paper):                     │
 * │                                                                        │
 * │    C(fX, b) = min Σᵢ ∫_{cell_i} |x - cᵢ|² · fX(x) dx               │
 * │                                                                        │
 * │  This is the Lloyd-Max algorithm applied to the Beta/Gaussian source. │
 * │  We precompute codebooks for b ∈ {1,2,3,4,8} and cache them here.    │
 * │                                                                        │
 * │  The paper specifies exact centroids for the Gaussian approximation   │
 * │  (large-d limit):                                                      │
 * │    b=1: {±√(2/π)/√d} ≈ {±0.7979/√d}                                 │
 * │    b=2: {±0.4528/√d, ±1.5104/√d}                                     │
 * │                                                                        │
 * │  For general dimensions d, the centroids are pre-normalised so that   │
 * │  the TurboQuant runtime scales them by ‖x‖₂ (the stored norm) at      │
 * │  dequantization, keeping the unit-sphere assumption harmless.          │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Implementation note:
 *   The centroids are stored for the standard normal N(0,1) distribution
 *   (the high-d Gaussian limit). At runtime the quantizer scales each
 *   coordinate by √d before looking up the centroid (since after rotation
 *   the coordinate variance is 1/d), so the effective quantization is
 *   applied to a N(0,1) variable.
 *
 *   For low dimensions (d ≤ 64), the Beta distribution deviates from the
 *   Gaussian, and the Gaussian centroids are sub-optimal. In that case
 *   the constructor accepts precomputed Beta centroids.
 */
public class LloydMaxCodebook {

    private static final Logger log = LoggerFactory.getLogger(LloydMaxCodebook.class);

    // ── Pre-computed Lloyd-Max centroids for N(0,1) ───────────────────────────
    // Obtained by running the Lloyd-Max algorithm to convergence on N(0,1).
    // Values agree with the Panter-Dite high-resolution formula for b ≥ 3.
    // Entries are sorted ascending; symmetric around 0 for odd/even 2^b.

    /** b=1: 2 centroids for N(0,1). Paper: ±√(2/π) ≈ ±0.7979 */
    private static final double[] CENTROIDS_1BIT = {
        -0.7978845608028654,
         0.7978845608028654
    };

    /** b=2: 4 centroids for N(0,1). Paper: ±0.4528, ±1.5104 */
    private static final double[] CENTROIDS_2BIT = {
        -1.5104176087566396,
        -0.4527789842943569,
         0.4527789842943569,
         1.5104176087566396
    };

    /** b=3: 8 centroids for N(0,1) (Lloyd-Max converged) */
    private static final double[] CENTROIDS_3BIT = {
        -2.1520032659509995,
        -1.3439796165053640,
        -0.7560052489539643,
        -0.2451619381626408,
         0.2451619381626408,
         0.7560052489539643,
         1.3439796165053640,
         2.1520032659509995
    };

    /** b=4: 16 centroids for N(0,1) (Lloyd-Max converged) */
    private static final double[] CENTROIDS_4BIT = {
        -2.7329629963774760,
        -2.0688299819863110,
        -1.6180460727688850,
        -1.2561274543497150,
        -0.9423029481456280,
        -0.6561221376010790,
        -0.3880241169447385,
        -0.1284225611143498,
         0.1284225611143498,
         0.3880241169447385,
         0.6561221376010790,
         0.9423029481456280,
         1.2561274543497150,
         1.6180460727688850,
         2.0688299819863110,
         2.7329629963774760
    };

    /** b=8: 256 centroids for N(0,1) (uniform spacing approximation for high b) */
    private static final double[] CENTROIDS_8BIT = buildUniformCentroids(8);

    // ── Instance fields ───────────────────────────────────────────────────────

    private final int bits;
    private final double[] centroids;  // sorted ascending, length = 2^bits
    private final float[]  centroidsF; // FP32 copy for fast SIMD lookup
    private final double   scale;      // scale factor: 1/√d normalisation

    /**
     * Creates a codebook for the given bit-width using the pre-computed
     * Gaussian (high-d) centroids, scaled for dimension d.
     *
     * @param bits       bit-width (1, 2, 3, 4, or 8)
     * @param dimension  vector dimension d (used to scale centroids by 1/√d)
     */
    public LloydMaxCodebook(int bits, int dimension) {
        this.bits  = bits;
        this.scale = 1.0 / Math.sqrt(dimension);
        this.centroids = loadCentroids(bits);

        // Scale the raw N(0,1) centroids by 1/√d for the Beta distribution
        // (each rotated coordinate has std ≈ 1/√d in the unit-sphere model)
        this.centroidsF = new float[centroids.length];
        for (int i = 0; i < centroids.length; i++) {
            centroidsF[i] = (float) (centroids[i] * scale);
        }

        log.debug("LloydMaxCodebook: bits={}, dim={}, levels={}, scale={}",
            bits, dimension, centroids.length, scale);
    }

    /**
     * Custom codebook constructor (e.g., pre-computed Beta centroids for low d).
     */
    public LloydMaxCodebook(double[] customCentroids) {
        this.bits = Integer.numberOfTrailingZeros(customCentroids.length);
        this.scale = 1.0;
        this.centroids = customCentroids.clone();
        this.centroidsF = new float[centroids.length];
        for (int i = 0; i < centroids.length; i++) centroidsF[i] = (float) centroids[i];
    }

    // ── Quantize / Dequantize ─────────────────────────────────────────────────

    /**
     * Quantizes a scalar value to the index of the nearest centroid.
     * Uses binary search since centroids are sorted ascending.
     *
     * @param value  the (rotated, scaled) coordinate value
     * @return       centroid index ∈ [0, 2^bits)
     */
    public int quantize(float value) {
        int lo = 0, hi = centroidsF.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            // Midpoint between centroid[mid] and centroid[mid+1]
            float boundary = (centroidsF[mid] + centroidsF[mid + 1]) * 0.5f;
            if (value <= boundary) hi = mid;
            else                   lo = mid + 1;
        }
        return lo;
    }

    /**
     * Scalar quantization over an entire coordinate array in-place.
     *
     * @param coords  rotated coordinate values, length = d
     * @param indices output quantization indices, length = d (pre-allocated)
     */
    public void quantize(float[] coords, int[] indices) {
        for (int i = 0; i < coords.length; i++) {
            indices[i] = quantize(coords[i]);
        }
    }

    /**
     * Dequantizes an index back to the centroid FP32 value.
     */
    public float dequantize(int index) {
        if (index < 0 || index >= centroidsF.length) return 0f;
        return centroidsF[index];
    }

    /**
     * Dequantizes a full index array to centroid values.
     *
     * @param indices  quantization indices, length = d
     * @param output   output centroid values, length = d (pre-allocated)
     */
    public void dequantize(int[] indices, float[] output) {
        for (int i = 0; i < indices.length; i++) {
            output[i] = dequantize(indices[i]);
        }
    }

    // ── Lloyd-Max Codebook Optimisation ───────────────────────────────────────

    /**
     * Refines an existing codebook using the Lloyd-Max iterative algorithm
     * applied to a set of sample values drawn from the source distribution.
     *
     * This is used when the pre-computed Gaussian centroids are a poor fit
     * (e.g., low d, or a very different input distribution).
     *
     * Algorithm:
     *   1. Assignment: assign each sample to the nearest centroid
     *   2. Update: recompute each centroid as the mean of its cluster
     *   3. Repeat until convergence (‖Δc‖ < tol)
     *
     * @param samples    representative coordinate samples (post-rotation)
     * @param maxIter    maximum Lloyd-Max iterations (default 100)
     * @param tol        convergence tolerance (default 1e-6)
     * @return           new refined LloydMaxCodebook
     */
    public static LloydMaxCodebook refine(
        double[] initialCentroids,
        double[] samples,
        int maxIter,
        double tol
    ) {
        int k = initialCentroids.length;
        double[] c = initialCentroids.clone();

        for (int iter = 0; iter < maxIter; iter++) {
            // Assignment step: accumulate sum and count per centroid
            double[] sum   = new double[k];
            int[]    count = new int[k];

            for (double s : samples) {
                int best = 0;
                double bestDist = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double d = Math.abs(s - c[j]);
                    if (d < bestDist) { bestDist = d; best = j; }
                }
                sum[best]   += s;
                count[best] += 1;
            }

            // Update step
            double maxDelta = 0;
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    double newC = sum[j] / count[j];
                    maxDelta = Math.max(maxDelta, Math.abs(newC - c[j]));
                    c[j] = newC;
                }
            }

            java.util.Arrays.sort(c); // keep sorted

            if (maxDelta < tol) {
                log.debug("Lloyd-Max converged at iteration {}", iter + 1);
                break;
            }
        }

        return new LloydMaxCodebook(c);
    }

    // ── Distortion Estimation ─────────────────────────────────────────────────

    /**
     * Estimates the scalar quantization MSE cost C(fX, b) using Monte Carlo.
     * Corresponds to Eq. (4) of the paper.
     *
     * @param samples  representative samples from the source distribution
     * @return         estimated per-coordinate MSE cost
     */
    public double estimateMseCost(float[] samples) {
        double mse = 0.0;
        for (float s : samples) {
            int idx = quantize(s);
            double diff = s - centroidsF[idx];
            mse += diff * diff;
        }
        return mse / samples.length;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public int getBits()        { return bits; }
    public int getNumLevels()   { return centroids.length; }
    public float[]  getCentroidsF()  { return centroidsF; }
    public double[] getCentroids()   { return centroids; }
    public double   getScale()       { return scale; }

    /**
     * Probability mass of each codebook entry under N(0,1).
     * Used for entropy coding bit-width reduction (§3.1 of paper).
     * The paper notes b=4 can be reduced by ~5% via Huffman coding.
     */
    public double[] computeEntropyCodingProbabilities() {
        double[] probs = new double[centroidsF.length];
        // Boundaries = midpoints between consecutive centroids
        double[] bounds = new double[centroidsF.length + 1];
        bounds[0]                    = Double.NEGATIVE_INFINITY;
        bounds[centroidsF.length]    = Double.POSITIVE_INFINITY;
        for (int i = 1; i < centroidsF.length; i++) {
            bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }

        // Integrate Gaussian PDF over each cell using CDF differences
        for (int i = 0; i < centroidsF.length; i++) {
            probs[i] = gaussianCdf(bounds[i + 1]) - gaussianCdf(bounds[i]);
        }
        return probs;
    }

    /**
     * Computes the Shannon entropy of the codebook's probability distribution.
     * The paper notes (§3.1): for b=4, entropy ≈ 3.8 bits (5% savings possible).
     */
    public double computeEntropy() {
        double[] probs = computeEntropyCodingProbabilities();
        double entropy = 0.0;
        for (double p : probs) {
            if (p > 1e-12) entropy -= p * Math.log(p) / Math.log(2);
        }
        return entropy;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private static double[] loadCentroids(int bits) {
        return switch (bits) {
            case 1  -> CENTROIDS_1BIT;
            case 2  -> CENTROIDS_2BIT;
            case 3  -> CENTROIDS_3BIT;
            case 4  -> CENTROIDS_4BIT;
            case 8  -> CENTROIDS_8BIT;
            default -> buildUniformCentroids(bits);
        };
    }

    /** Builds uniform centroids for high bit-widths (Panter-Dite approximation). */
    private static double[] buildUniformCentroids(int bits) {
        int k = 1 << bits;
        double[] c = new double[k];
        // Uniform quantizer on [-4σ, +4σ] range (captures 99.99% of N(0,1))
        double range = 8.0;  // [-4, +4]
        double step  = range / k;
        for (int i = 0; i < k; i++) {
            c[i] = -4.0 + step * (i + 0.5);
        }
        return c;
    }

    /** Standard normal CDF using erfc approximation (Horner's method) */
    private static double gaussianCdf(double x) {
        if (x == Double.NEGATIVE_INFINITY) return 0.0;
        if (x == Double.POSITIVE_INFINITY) return 1.0;
        return 0.5 * erfc(-x / Math.sqrt(2));
    }

    private static double erfc(double x) {
        // Abramowitz & Stegun approximation (max error 1.5e-7)
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        double result = poly * Math.exp(-x * x);
        return x >= 0 ? result : 2.0 - result;
    }

    @Override
    public String toString() {
        return "LloydMaxCodebook{bits=%d, levels=%d, scale=%.4f, entropy=%.3f bits}"
            .formatted(bits, centroids.length, scale, computeEntropy());
    }
}
