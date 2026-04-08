package tech.kayys.gollek.spi.tensor;

/**
 * Service Provider Interface for hardware-accelerated tensor computation.
 *
 * <p>
 * Implementations of this interface bridge the SDK's autograd engine to
 * hardware-specific kernel plugins (CUDA, Metal, ROCm, etc.).
 * The default implementation ({@link CpuBackend}) runs all operations
 * in pure Java on the CPU.
 *
 * <p>
 * Custom backends are discovered via {@link java.util.ServiceLoader}.
 * Register implementations in
 * {@code META-INF/services/tech.kayys.gollek.spi.tensor.ComputeBackend}.
 *
 * @see CpuBackend
 * @see ComputeBackendRegistry
 */
public interface ComputeBackend {

    /**
     * Matrix multiplication: C = A @ B.
     *
     * @param a      flat data for matrix A in row-major order
     * @param shapeA shape of A (e.g., [m, k])
     * @param b      flat data for matrix B in row-major order
     * @param shapeB shape of B (e.g., [k, n])
     * @return flat result array in row-major order with shape [m, n]
     */
    float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB);

    /**
     * Element-wise addition: C = A + B.
     *
     * @param a     flat data for tensor A
     * @param b     flat data for tensor B
     * @param shape shared shape of A and B
     * @return element-wise sum
     */
    float[] add(float[] a, float[] b, long[] shape);

    /**
     * Element-wise subtraction: C = A - B.
     */
    float[] sub(float[] a, float[] b, long[] shape);

    /**
     * Element-wise multiplication (Hadamard product): C = A ⊙ B.
     */
    float[] mul(float[] a, float[] b, long[] shape);

    /**
     * Element-wise division: C = A / B.
     */
    float[] div(float[] a, float[] b, long[] shape);

    /**
     * ReLU activation: max(0, x) element-wise.
     */
    float[] relu(float[] data, long[] shape);

    /**
     * Sigmoid activation: 1 / (1 + exp(-x)) element-wise.
     */
    float[] sigmoid(float[] data, long[] shape);

    /**
     * Tanh activation element-wise.
     */
    float[] tanh(float[] data, long[] shape);

    /**
     * Exponential: exp(x) element-wise.
     */
    float[] exp(float[] data, long[] shape);

    /**
     * Natural logarithm: ln(x) element-wise.
     */
    float[] log(float[] data, long[] shape);

    /**
     * Sum all elements to a single scalar.
     *
     * @return array of length 1 containing the sum
     */
    float[] sum(float[] data, long[] shape);

    /**
     * Mean of all elements.
     *
     * @return array of length 1 containing the mean
     */
    float[] mean(float[] data, long[] shape);

    /**
     * Transpose a 2D matrix.
     */
    float[] transpose2d(float[] data, long rows, long cols);

    /**
     * Power: x^p element-wise.
     */
    float[] pow(float[] data, long[] shape, float p);

    /**
     * Returns the device this backend targets.
     */
    String deviceName();

    /**
     * Priority for backend selection. Higher = preferred.
     * CPU backend returns 0; GPU backends should return higher values.
     */
    default int priority() {
        return 0;
    }
}
