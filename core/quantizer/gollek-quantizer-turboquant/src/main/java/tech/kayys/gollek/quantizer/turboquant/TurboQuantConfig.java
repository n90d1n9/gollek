package tech.kayys.gollek.quantizer.turboquant;

/**
 * Configuration for TurboQuant (Zandieh et al., 2025).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  TurboQuant: Online Vector Quantization with Near-optimal Distortion   │
 * │  arXiv:2504.19874v1                                                    │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  Two variants (§3 of the paper):                                       │
 * │                                                                        │
 * │  TurboQuant_mse  (Algorithm 1):                                        │
 * │    • Minimises MSE: Dmse = E[‖x − x̃‖²]                               │
 * │    • Randomly rotate x → Π·x  (uniform on unit sphere → Beta coords) │
 * │    • Apply optimal Lloyd-Max scalar quantizer per coordinate           │
 * │    • Dequant: retrieve centroids, rotate back  x̃ = Πᵀ·ỹ             │
 * │    • MSE bound: Dmse ≤ (√(3π)/2) · 4^(-b) ≈ 2.72 · 4^(-b)          │
 * │    • BIASED for inner products (bias = 2/π at b=1)                    │
 * │                                                                        │
 * │  TurboQuant_prod (Algorithm 2):                                        │
 * │    • Unbiased inner product: E[⟨y, x̃⟩] = ⟨y, x⟩                    │
 * │    • Stage 1: apply Qmse with bit-width (b-1)                         │
 * │    • Compute residual r = x − x̃_mse                                  │
 * │    • Stage 2: apply QJL on residual: sign(S·r), S ~ N(0,1)^(d×d)    │
 * │    • Store: (idx, qjl, ‖r‖₂)                                          │
 * │    • Dequant: x̃ = x̃_mse + (√(π/2)/d) · ‖r‖ · Sᵀ · qjl            │
 * │    • Inner prod error: Dprod ≤ (√(3π)/2) · ‖y‖²/d · 4^(-b)         │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * KV Cache mode (§4.2-4.3 of paper):
 *   Applies TurboQuant_prod online during transformer forward pass.
 *   Outlier channels are quantized at higher bit-width:
 *     32 outlier channels at 3 bits + 96 normal at 2 bits → 2.5 effective bits.
 *
 * Rotation strategy:
 *   The paper uses a full d×d random Gaussian rotation (QR decomposition).
 *   For practical implementations we support the fast Walsh-Hadamard Transform
 *   (WHT/FWHT) composed with random sign-flipping (DiagSign · H · DiagSign),
 *   which is equivalent in distribution to a uniformly random rotation and
 *   runs in O(d log d) instead of O(d²). This is the approach used by QuaRot
 *   and related works.
 */
public record TurboQuantConfig(
    /** Bit-width b per coordinate (1, 2, 3, 4, or 8) */
    int bits,

    /** Dimension of the vectors to quantize */
    int dimension,

    /** Which variant to use */
    Variant variant,

    /** Rotation strategy to apply before scalar quantization */
    RotationStrategy rotation,

    /**
     * Whether to use outlier-channel splitting for KV cache.
     * If true, the top-{outlierChannels} channels are quantized at (bits+1)-bits.
     */
    boolean splitOutliers,

    /** Number of outlier channels when splitOutliers=true */
    int outlierChannels,

    /** Random seed for reproducible rotation matrices */
    long seed
) {

    /** TurboQuant algorithm variant */
    public enum Variant {
        /**
         * MSE-optimal (Algorithm 1).
         * Minimises E[‖x − x̃‖²]. Biased for inner products at low bit-widths.
         */
        MSE,

        /**
         * Inner-product-optimal (Algorithm 2).
         * Unbiased estimator for ⟨y,x⟩. Uses (b-1) bits for Qmse + 1 bit QJL residual.
         * Requires one extra bit overhead vs MSE variant for same IP quality.
         */
        INNER_PRODUCT
    }

    /** Strategy for the random rotation Π before scalar quantization */
    public enum RotationStrategy {
        /**
         * Full random orthogonal rotation via QR decomposition.
         * Exact uniform random rotation. Cost: O(d²) storage + O(d²) apply.
         * Suitable for small-to-medium dimensions (d ≤ 4096).
         */
        RANDOM_ORTHOGONAL,

        /**
         * Fast Walsh-Hadamard Transform with random diagonal sign-flips.
         * Π = D₂ · H · D₁ where D₁, D₂ are random {±1} diagonal matrices and H is WHT.
         * Equivalent in expectation to uniform random rotation.
         * Cost: O(d log d) apply, O(d) storage for sign vectors.
         * Preferred for large d (attention head dimensions in LLMs: 128, 256, 512).
         */
        HADAMARD,

        /**
         * Randomized SVD-based rotation (structured random matrix).
         * Uses a smaller random matrix projected to orthogonal.
         * Intermediate cost/quality tradeoff.
         */
        RANDOM_SVD
    }

    // ── Preset Configurations ─────────────────────────────────────────────────

    /** TurboQuant_mse, 4-bit, Hadamard rotation (recommended for weight quantization) */
    public static TurboQuantConfig mse4bit(int dimension) {
        return new TurboQuantConfig(4, dimension, Variant.MSE,
            RotationStrategy.HADAMARD, false, 0, 42L);
    }

    /** TurboQuant_prod, 4-bit, Hadamard (recommended for KV cache at 3.5-bit effective) */
    public static TurboQuantConfig prod4bit(int dimension) {
        return new TurboQuantConfig(4, dimension, Variant.INNER_PRODUCT,
            RotationStrategy.HADAMARD, false, 0, 42L);
    }

    /** TurboQuant_prod, 2-bit, Hadamard (KV cache 2.5-bit effective with outlier split) */
    public static TurboQuantConfig prod2bitKvCache(int dimension) {
        return new TurboQuantConfig(2, dimension, Variant.INNER_PRODUCT,
            RotationStrategy.HADAMARD, true, 32, 42L);
    }

    /** TurboQuant_prod, 3-bit, Hadamard (3.5-bit effective with outlier split) */
    public static TurboQuantConfig prod3bitKvCache(int dimension) {
        return new TurboQuantConfig(3, dimension, Variant.INNER_PRODUCT,
            RotationStrategy.HADAMARD, true, 32, 42L);
    }

    /** Full random orthogonal rotation, MSE, 4-bit (highest quality, slower) */
    public static TurboQuantConfig mse4bitExact(int dimension) {
        return new TurboQuantConfig(4, dimension, Variant.MSE,
            RotationStrategy.RANDOM_ORTHOGONAL, false, 0, 42L);
    }

    // ── Derived Properties ────────────────────────────────────────────────────

    /** Number of quantization levels = 2^bits */
    public int numLevels() { return 1 << bits; }

    /**
     * Effective bits-per-coordinate for the MSE stage of Qprod.
     * Qprod uses Qmse at (bits-1) bits, then 1-bit QJL on residual.
     */
    public int mseStageBits() {
        return variant == Variant.INNER_PRODUCT ? bits - 1 : bits;
    }

    /**
     * Upper bound on MSE distortion from Theorem 1 of the paper.
     * Dmse ≤ (√(3π)/2) · 4^(-b)
     */
    public double mseBound() {
        return Math.sqrt(3 * Math.PI) / 2.0 * Math.pow(4.0, -bits);
    }

    /**
     * Empirical MSE for b ∈ {1,2,3,4} from paper (Theorem 1, fine-grained):
     * b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009
     */
    public double empiricalMse() {
        return switch (bits) {
            case 1 -> 0.36;
            case 2 -> 0.117;
            case 3 -> 0.030;
            case 4 -> 0.009;
            default -> mseBound();
        };
    }

    /**
     * Upper bound on inner product distortion per Theorem 2 (‖y‖²/d factor excluded):
     * Dprod ≤ (√(3π)/2) · 4^(-b)
     */
    public double innerProductBound() {
        return Math.sqrt(3 * Math.PI) / 2.0 * Math.pow(4.0, -bits);
    }

    /**
     * Information-theoretic lower bound from Theorem 3: Dmse ≥ 4^(-b)
     */
    public double lowerBound() { return Math.pow(4.0, -bits); }

    /**
     * Ratio of TurboQuant MSE bound to information-theoretic lower bound.
     * Paper proves this is ≤ √(3π)/2 ≈ 2.72 for all b.
     */
    public double optimalityGap() { return Math.sqrt(3 * Math.PI) / 2.0; }

    /**
     * Effective bits per coordinate including outlier channel overhead.
     * From paper §4.3: 2.5-bit = (32×3 + 96×2)/128
     */
    public double effectiveBitsPerChannel(int totalChannels) {
        if (!splitOutliers || outlierChannels == 0) return bits;
        int normal = totalChannels - outlierChannels;
        return (double) (outlierChannels * (bits + 1) + normal * bits) / totalChannels;
    }

    public boolean isMseVariant()          { return variant == Variant.MSE; }
    public boolean isInnerProductVariant() { return variant == Variant.INNER_PRODUCT; }
    public boolean usesHadamard()          { return rotation == RotationStrategy.HADAMARD; }

    @Override
    public String toString() {
        return "TurboQuantConfig{bits=%d, dim=%d, variant=%s, rot=%s, seed=%d}"
            .formatted(bits, dimension, variant, rotation, seed);
    }
}
