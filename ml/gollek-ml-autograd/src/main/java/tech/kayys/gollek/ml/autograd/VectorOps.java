package tech.kayys.gollek.ml.autograd;

import java.util.Objects;

/**
 * Small vector helpers used throughout the legacy ML/trainer surface.
 */
public final class VectorOps {

    private VectorOps() {
    }

    public static float sum(float[] values) {
        float result = 0f;
        for (float value : values) {
            result += value;
        }
        return result;
    }

    public static float max(float[] values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("values must not be empty");
        }
        float result = values[0];
        for (int i = 1; i < values.length; i++) {
            result = Math.max(result, values[i]);
        }
        return result;
    }

    public static void mul(float[] left, float[] right, float[] out) {
        checkSameLength(left, right, out);
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] * right[i];
        }
    }

    public static void mulScalar(float[] input, float scalar, float[] out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");
        if (input.length != out.length) {
            throw new IllegalArgumentException("input and out must have same length");
        }
        for (int i = 0; i < out.length; i++) {
            out[i] = input[i] * scalar;
        }
    }

    public static void add(float[] left, float[] right, float[] out) {
        checkSameLength(left, right, out);
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] + right[i];
        }
    }

    public static void addScalar(float[] input, float scalar, float[] out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");
        if (input.length != out.length) {
            throw new IllegalArgumentException("input and out must have same length");
        }
        for (int i = 0; i < out.length; i++) {
            out[i] = input[i] + scalar;
        }
    }

    /**
     * Dense row-major matrix multiplication:
     * left [m, n] x right [n, p] -> out [m, p]
     */
    public static float[] matmul(float[] left, float[] right, int m, int n, int p) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        if (left.length != m * n) {
            throw new IllegalArgumentException("left length does not match matrix shape");
        }
        if (right.length != n * p) {
            throw new IllegalArgumentException("right length does not match matrix shape");
        }

        float[] out = new float[m * p];
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < p; col++) {
                float acc = 0f;
                for (int k = 0; k < n; k++) {
                    acc += left[row * n + k] * right[k * p + col];
                }
                out[row * p + col] = acc;
            }
        }
        return out;
    }

    private static void checkSameLength(float[] left, float[] right, float[] out) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(out, "out");
        if (left.length != right.length || left.length != out.length) {
            throw new IllegalArgumentException("left, right and out must have same length");
        }
    }
}
