package tech.kayys.gollek.ml.autograd;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.ComputeBackendRegistry;
import tech.kayys.gollek.spi.tensor.CpuBackend;

/**
 * Hardware-aware tensor operations dispatcher.
 *
 * <p>Prioritizes delegation to the active {@link ComputeBackend} (e.g., Metal or CUDA).
 * Falls back to SIMD-accelerated Java (using JDK 25 Vector API) when running on CPU.
 */
public final class VectorOps {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private VectorOps() {}

    // ── Element-wise ─────────────────────────────────────────────────────

    public static float[] add(float[] a, float[] b, float[] out) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            float[] res = backend.add(a, b, new long[]{a.length});
            System.arraycopy(res, 0, out, 0, a.length);
            return out;
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .add(FloatVector.fromArray(SPECIES, b, i))
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] + b[i];
        return out;
    }

    public static float[] sub(float[] a, float[] b, float[] out) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            float[] res = backend.sub(a, b, new long[]{a.length});
            System.arraycopy(res, 0, out, 0, a.length);
            return out;
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .sub(FloatVector.fromArray(SPECIES, b, i))
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] - b[i];
        return out;
    }

    public static float[] mul(float[] a, float[] b, float[] out) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            float[] res = backend.mul(a, b, new long[]{a.length});
            System.arraycopy(res, 0, out, 0, a.length);
            return out;
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .mul(FloatVector.fromArray(SPECIES, b, i))
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] * b[i];
        return out;
    }

    public static float[] div(float[] a, float[] b, float[] out) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            float[] res = backend.div(a, b, new long[]{a.length});
            System.arraycopy(res, 0, out, 0, a.length);
            return out;
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .div(FloatVector.fromArray(SPECIES, b, i))
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] / b[i];
        return out;
    }

    public static float[] relu(float[] a, float[] out) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            float[] res = backend.relu(a, new long[]{a.length});
            System.arraycopy(res, 0, out, 0, a.length);
            return out;
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .max(zero)
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = Math.max(0f, a[i]);
        return out;
    }

    public static float[] exp(float[] a, float[] out) {
        // Vector API doesn't have exp natively — use scalar but keep loop structure
        for (int i = 0; i < a.length; i++) out[i] = (float) Math.exp(a[i]);
        return out;
    }

    public static float[] tanh(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = (float) Math.tanh(a[i]);
        return out;
    }

    public static float[] sigmoid(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = 1f / (1f + (float) Math.exp(-a[i]));
        return out;
    }

    public static float[] log(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = (float) Math.log(a[i] + 1e-8f);
        return out;
    }

    public static float[] pow(float[] a, float p, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = (float) Math.pow(a[i], p);
        return out;
    }

    public static float[] abs(float[] a, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).abs().intoArray(out, i);
        }
        for (; i < len; i++) out[i] = Math.abs(a[i]);
        return out;
    }

    public static float[] sqrt(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = (float) Math.sqrt(a[i]);
        return out;
    }

    public static float[] rsqrt(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) out[i] = (float) (1.0 / Math.sqrt(a[i]));
        return out;
    }

    public static float[] clamp(float[] a, float min, float max, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector minV = FloatVector.broadcast(SPECIES, min);
        FloatVector maxV = FloatVector.broadcast(SPECIES, max);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).max(minV).min(maxV).intoArray(out, i);
        }
        for (; i < len; i++) out[i] = Math.max(min, Math.min(max, a[i]));
        return out;
    }

    public static float[] neg(float[] a, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).neg().intoArray(out, i);
        }
        for (; i < len; i++) out[i] = -a[i];
        return out;
    }

    public static float[] sigmoidGrad(float[] a, float[] out) {
        for (int i = 0; i < a.length; i++) {
            float s = 1f / (1f + (float) Math.exp(-a[i]));
            out[i] = s * (1 - s);
        }
        return out;
    }

    // ── Reductions ───────────────────────────────────────────────────────

    public static float sum(float[] a) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            return backend.sum(a, new long[]{a.length})[0];
        }

        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector acc = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            acc = acc.add(FloatVector.fromArray(SPECIES, a, i));
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) result += a[i];
        return result;
    }

    public static float max(float[] a) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector acc = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
        for (; i < bound; i += SPECIES.length()) {
            acc = acc.max(FloatVector.fromArray(SPECIES, a, i));
        }
        float result = acc.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) result = Math.max(result, a[i]);
        return result;
    }

    // ── Scalar broadcast ─────────────────────────────────────────────────

    public static float[] mulScalar(float[] a, float s, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector sv = FloatVector.broadcast(SPECIES, s);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).mul(sv).intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] * s;
        return out;
    }

    public static float[] addScalar(float[] a, float s, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        FloatVector sv = FloatVector.broadcast(SPECIES, s);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).add(sv).intoArray(out, i);
        }
        for (; i < len; i++) out[i] = a[i] + s;
        return out;
    }

    // ── Matrix multiply (row-major, 2D) ──────────────────────────────────

    /**
     * Computes C = A @ B where A is [M x K] and B is [K x N], row-major.
     * Uses Vector API for the inner dot-product loop.
     */
    public static float[] matmul(float[] a, float[] b, int M, int K, int N) {
        ComputeBackend backend = ComputeBackendRegistry.get();
        if (!(backend instanceof CpuBackend)) {
            return backend.matmul(a, new long[]{M, K}, b, new long[]{K, N});
        }

        float[] c = new float[M * N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int i = 0;
                int bound = SPECIES.loopBound(K);
                FloatVector acc = FloatVector.zero(SPECIES);
                for (; i < bound; i += SPECIES.length()) {
                    FloatVector av = FloatVector.fromArray(SPECIES, a, m * K + i);
                    // Gather column n of B: B[(i+lane)*N + n]
                    FloatVector bv = FloatVector.fromArray(SPECIES, b, n,
                        buildStridedIndex(i, N, SPECIES.length()), 0);
                    acc = acc.add(av.mul(bv));
                }
                float dot = acc.reduceLanes(VectorOperators.ADD);
                for (; i < K; i++) dot += a[m * K + i] * b[i * N + n];
                c[m * N + n] = dot;
            }
        }
        return c;
    }

    /** Build index array for strided gather (column of row-major matrix). */
    private static int[] buildStridedIndex(int base, int stride, int len) {
        int[] idx = new int[len];
        for (int i = 0; i < len; i++) idx[i] = (base + i) * stride;
        return idx;
    }

    // ── Fused multiply-add ───────────────────────────────────────────────

    /** out[i] = a[i] * b[i] + c[i]  (FMA) */
    public static float[] fma(float[] a, float[] b, float[] c, float[] out) {
        int i = 0, len = a.length;
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i)
                .fma(FloatVector.fromArray(SPECIES, b, i),
                     FloatVector.fromArray(SPECIES, c, i))
                .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = Math.fma(a[i], b[i], c[i]);
        return out;
    }
}
