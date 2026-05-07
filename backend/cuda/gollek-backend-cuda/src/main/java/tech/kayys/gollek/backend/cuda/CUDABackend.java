package tech.kayys.gollek.backend.cuda;

import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Tensor;
import java.util.List;

/**
 * CUDA compute backend using FFM API for GPU-accelerated tensor operations.
 * Method bodies delegate to native CUDA kernels via downcall handles;
 * unimplemented kernels throw {@link UnsupportedOperationException}.
 */
public final class CUDABackend implements ComputeBackend {

    private static final String TODO = "CUDA kernel not yet bound via FFM";

    @Override
    public Tensor add(Tensor a, Tensor b) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor softmax(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor mean(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor abs(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor cast(Tensor a, DType dtype) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor to(Tensor a, DeviceType device) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }
}