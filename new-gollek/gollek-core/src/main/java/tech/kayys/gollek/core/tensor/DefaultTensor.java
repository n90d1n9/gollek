package tech.kayys.gollek.core.tensor;

import tech.kayys.gollek.core.memory.Buffer;
import tech.kayys.gollek.core.backend.ComputeBackend;

public final class DefaultTensor implements Tensor {
    private final Shape shape;
    private final DType dtype;
    private final Device device;
    private final Buffer buffer;
    private final ComputeBackend backend;

    public DefaultTensor(
            Shape shape,
            DType dtype,
            Device device,
            Buffer buffer,
            ComputeBackend backend) {
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
        this.buffer = buffer;
        this.backend = backend;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public Device device() {
        return device;
    }

    public Buffer buffer() {
        return buffer;
    }

    @Override
    public Tensor mul(Tensor other) {
        throw new UnsupportedOperationException("T x T mul not implemented in DefaultTensor yet");
    }

    @Override
    public Tensor add(Tensor other) {
        return backend.add(this, other);
    }

    @Override
    public Tensor mul(float scalar) {
        return backend.mul(this, scalar);
    }

    @Override
    public Tensor matmul(Tensor other) {
        return backend.matmul(this, other);
    }

    @Override
    public Tensor reshape(long... newShape) {
        return backend.reshape(this, newShape);
    }
}