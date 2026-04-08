package tech.kayys.gollek.spi.tensor;

/**
 * Default pure-Java CPU implementation of {@link ComputeBackend}.
 *
 * <p>All operations run on the JVM heap using standard Java arrays.
 * This backend is always available and serves as the fallback when
 * no hardware-accelerated backend is on the classpath.
 */
public final class CpuBackend implements ComputeBackend {

    @Override
    public float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB) {
        int m = (int) shapeA[0];
        int k = (int) shapeA[1];
        int n = (int) shapeB[1];
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    @Override
    public float[] add(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    @Override
    public float[] sub(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    @Override
    public float[] mul(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] * b[i];
        }
        return c;
    }

    @Override
    public float[] div(float[] a, float[] b, long[] shape) {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] / b[i];
        }
        return c;
    }

    @Override
    public float[] relu(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.max(0, data[i]);
        }
        return result;
    }

    @Override
    public float[] sigmoid(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) (1.0 / (1.0 + Math.exp(-data[i])));
        }
        return result;
    }

    @Override
    public float[] tanh(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.tanh(data[i]);
        }
        return result;
    }

    @Override
    public float[] exp(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.exp(data[i]);
        }
        return result;
    }

    @Override
    public float[] log(float[] data, long[] shape) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.log(data[i]);
        }
        return result;
    }

    @Override
    public float[] sum(float[] data, long[] shape) {
        float s = 0;
        for (float v : data) s += v;
        return new float[]{s};
    }

    @Override
    public float[] mean(float[] data, long[] shape) {
        float s = 0;
        for (float v : data) s += v;
        return new float[]{s / data.length};
    }

    @Override
    public float[] transpose2d(float[] data, long rows, long cols) {
        int r = (int) rows, c = (int) cols;
        float[] result = new float[data.length];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                result[j * r + i] = data[i * c + j];
            }
        }
        return result;
    }

    @Override
    public float[] pow(float[] data, long[] shape, float p) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.pow(data[i], p);
        }
        return result;
    }

    @Override
    public String deviceName() {
        return "CPU";
    }

    @Override
    public int priority() {
        return 0; // lowest — any GPU backend takes precedence
    }
}
