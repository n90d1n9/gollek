package tech.kayys.gollek.ml.tensor;

/**
 * Deprecated: use {@link tech.kayys.gollek.ml.autograd.VectorOps} instead.
 * This class exists for backward compatibility only.
 *
 * @deprecated Use {@code tech.kayys.gollek.ml.autograd.VectorOps}
 */
@Deprecated(since = "0.1.0", forRemoval = true)
public final class VectorOps {
    private VectorOps() {}

    public static float[] add(float[] a, float[] b, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.add(a, b, out);
    }

    public static float[] sub(float[] a, float[] b, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.sub(a, b, out);
    }

    public static float[] mul(float[] a, float[] b, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.mul(a, b, out);
    }

    public static float[] div(float[] a, float[] b, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.div(a, b, out);
    }

    public static float[] relu(float[] a, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.relu(a, out);
    }

    public static float[] exp(float[] a, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.exp(a, out);
    }

    public static float[] tanh(float[] a, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.tanh(a, out);
    }

    public static float sum(float[] a) {
        return tech.kayys.gollek.ml.autograd.VectorOps.sum(a);
    }

    public static float max(float[] a) {
        return tech.kayys.gollek.ml.autograd.VectorOps.max(a);
    }

    public static float[] mulScalar(float[] a, float s, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.mulScalar(a, s, out);
    }

    public static float[] addScalar(float[] a, float s, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.addScalar(a, s, out);
    }

    public static float[] matmul(float[] a, float[] b, int M, int K, int N) {
        return tech.kayys.gollek.ml.autograd.VectorOps.matmul(a, b, M, K, N);
    }

    public static float[] fma(float[] a, float[] b, float[] c, float[] out) {
        return tech.kayys.gollek.ml.autograd.VectorOps.fma(a, b, c, out);
    }
}
