package tech.kayys.gollek.runtime.inference.kv;

import org.jboss.logging.Logger;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Local TurboQuant engine for runtime-inference module.
 * Avoids cyclic dependency with gollek-quantizer-turboquant.
 * 
 * This is a simplified implementation focused on KV cache operations,
 * providing quantization and dequantization using TurboQuant algorithm.
 */
public final class TurboQuantLocalEngine {

    private static final Logger LOG = Logger.getLogger(TurboQuantLocalEngine.class);

    // SIMD vector species
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    /** Configuration */
    private final TurboQuantTypes.TurboQuantConfig config;
    private final int dim;

    /** Precomputed Hadamard rotation seed */
    private final long seed;

    /**
     * Creates a new TurboQuant local engine.
     *
     * @param config TurboQuant configuration
     */
    public TurboQuantLocalEngine(TurboQuantTypes.TurboQuantConfig config) {
        this.config = config;
        this.dim = config.dimension();
        this.seed = config.seed();

        LOG.infof("TurboQuantLocalEngine: dim=%d, bits=%d, SIMD lanes=%d",
            dim, config.bits(), F_LANES);
    }

    // ── Quantization ───────────────────────────────────────────────────

    /**
     * Quantizes a vector using TurboQuant prod algorithm.
     *
     * @param vector input vector [dim]
     * @return quantized result with indices, signs, and residual norm
     */
    public TurboQuantTypes.QuantProdResult quantizeProd(float[] vector) {
        if (vector.length != dim) {
            throw new IllegalArgumentException(
                "Vector dimension mismatch: expected " + dim + ", got " + vector.length);
        }

        int bits = config.bits();
        int mseBits = bits - 1;
        int codebookSize = 1 << mseBits;

        // Apply rotation (simplified Hadamard)
        float[] rotated = applyHadamard(vector);

        // MSE quantization: find nearest centroids
        int[] mseIndices = new int[dim];
        float[] dequantized = new float[dim];
        float[] codebook = buildCodebook(mseBits);

        for (int i = 0; i < dim; i++) {
            // Find nearest centroid
            int bestIdx = 0;
            float bestDist = Float.MAX_VALUE;
            for (int c = 0; c < codebookSize; c++) {
                float dist = Math.abs(rotated[i] - codebook[c]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = c;
                }
            }
            mseIndices[i] = bestIdx;
            dequantized[i] = codebook[bestIdx];
        }

        // Compute residual
        float[] residual = new float[dim];
        float residualNormSq = 0;
        for (int i = 0; i < dim; i++) {
            residual[i] = vector[i] - dequantized[i];
            residualNormSq += residual[i] * residual[i];
        }
        float residualNorm = (float) Math.sqrt(residualNormSq);

        // QJL stage: sign of random projection (simplified)
        byte[] qjlSigns = computeQjlSigns(residual, residualNorm);

        return new TurboQuantTypes.QuantProdResult(mseIndices, qjlSigns, residualNorm);
    }

    // ── Dequantization ─────────────────────────────────────────────────

    /**
     * Dequantizes a quantized result back to float vector.
     *
     * @param result quantized result
     * @param output output vector [dim]
     */
    public void dequantizeProd(TurboQuantTypes.QuantProdResult result, float[] output) {
        if (output.length != dim) {
            throw new IllegalArgumentException(
                "Output dimension mismatch: expected " + dim + ", got " + output.length);
        }

        int mseBits = config.bits() - 1;
        int codebookSize = 1 << mseBits;
        float[] codebook = buildCodebook(mseBits);

        // Reconstruct from MSE indices
        float[] mseReconstructed = new float[dim];
        for (int i = 0; i < dim; i++) {
            mseReconstructed[i] = codebook[result.mseIndices()[i]];
        }

        // Add QJL residual correction
        if (result.residualNorm() > 0) {
            float[] qjlCorrection = reconstructQjl(result);
            for (int i = 0; i < dim; i++) {
                mseReconstructed[i] += qjlCorrection[i];
            }
        }

        // Apply inverse rotation
        applyInverseHadamard(mseReconstructed, output);
    }

    /**
     * Estimates inner product between a query and a quantized vector.
     * This is the key operation for attention without full dequantization.
     *
     * @param query query vector [dim]
     * @param result quantized key/value
     * @return estimated inner product
     */
    public float estimateInnerProductFull(float[] query, TurboQuantTypes.QuantProdResult result) {
        if (query.length != dim) {
            throw new IllegalArgumentException(
                "Query dimension mismatch: expected " + dim + ", got " + query.length);
        }

        // Stage 1: MSE part
        int mseBits = config.bits() - 1;
        int codebookSize = 1 << mseBits;
        float[] codebook = buildCodebook(mseBits);

        float mseInnerProduct = 0;
        for (int i = 0; i < dim; i++) {
            mseInnerProduct += query[i] * codebook[result.mseIndices()[i]];
        }

        // Stage 2: QJL residual correction
        if (result.residualNorm() > 0) {
            float qjlCorrection = estimateQjlInnerProduct(query, result);
            mseInnerProduct += qjlCorrection;
        }

        return mseInnerProduct;
    }

    // ── Internal Helpers ───────────────────────────────────────────────

    /**
     * Builds a Lloyd-Max codebook for given bit width (simplified).
     * For production, this should use precomputed optimal centroids.
     */
    private float[] buildCodebook(int bits) {
        int size = 1 << bits;
        float[] codebook = new float[size];

        // Use uniform quantizer for N(0, 1/d) as approximation
        float std = 1.0f / (float) Math.sqrt(dim);
        float range = std * 4;  // Cover ±4σ
        float step = (2 * range) / size;

        for (int i = 0; i < size; i++) {
            codebook[i] = -range + (i + 0.5f) * step;
        }

        return codebook;
    }

    /**
     * Applies simplified Hadamard rotation.
     */
    private float[] applyHadamard(float[] vector) {
        // Simplified: just copy (full Hadamard would be O(d log d))
        float[] result = new float[dim];
        System.arraycopy(vector, 0, result, 0, dim);
        return result;
    }

    /**
     * Applies inverse Hadamard rotation.
     */
    private void applyInverseHadamard(float[] input, float[] output) {
        // Simplified: just copy
        System.arraycopy(input, 0, output, 0, dim);
    }

    /**
     * Computes QJL signs for residual vector.
     */
    private byte[] computeQjlSigns(float[] residual, float norm) {
        byte[] signs = new byte[dim];
        if (norm > 0) {
            // Use pseudo-random projections (deterministic from seed)
            java.util.Random rng = new java.util.Random(seed);
            for (int i = 0; i < dim; i++) {
                float projection = rng.nextFloat() * 2 - 1;  // [-1, 1]
                signs[i] = (byte) (projection >= 0 ? 1 : -1);
            }
        }
        return signs;
    }

    /**
     * Reconstructs QJL residual correction.
     */
    private float[] reconstructQjl(TurboQuantTypes.QuantProdResult result) {
        float[] correction = new float[dim];
        if (result.residualNorm() > 0) {
            java.util.Random rng = new java.util.Random(seed);
            float scale = result.residualNorm() / (float) Math.sqrt(dim);
            for (int i = 0; i < dim; i++) {
                float projection = rng.nextFloat() * 2 - 1;
                correction[i] = scale * (projection >= 0 ? 1 : -1) * result.qjlSigns()[i];
            }
        }
        return correction;
    }

    /**
     * Estimates QJL inner product contribution.
     */
    private float estimateQjlInnerProduct(float[] query, TurboQuantTypes.QuantProdResult result) {
        if (result.residualNorm() == 0) return 0;

        float sum = 0;
        java.util.Random rng = new java.util.Random(seed);
        float scale = result.residualNorm() / (float) Math.sqrt(dim);

        for (int i = 0; i < dim; i++) {
            float projection = rng.nextFloat() * 2 - 1;
            sum += query[i] * scale * (projection >= 0 ? 1 : -1) * result.qjlSigns()[i];
        }

        return sum;
    }

    // ── Getters ────────────────────────────────────────────────────────

    public TurboQuantTypes.TurboQuantConfig getConfig() {
        return config;
    }

    public int dimension() {
        return dim;
    }
}
