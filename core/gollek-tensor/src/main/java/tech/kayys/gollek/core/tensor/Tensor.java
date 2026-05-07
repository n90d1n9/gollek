package tech.kayys.gollek.core.tensor;


import tech.kayys.gollek.core.backend.ComputeBackend;

public interface Tensor {
    Shape shape();

    DeviceType device();

    DType dtype();

    ComputeBackend backend();

    Tensor add(Tensor other);

    Tensor sub(Tensor other);

    Tensor mul(Tensor other);

    Tensor mul(float scalar);

    Tensor div(float scalar);

    Tensor matmul(Tensor other);

    Tensor reshape(long... newShape);

    Tensor softmax();

    Tensor slice(long[] offsets, long[] sizes);

    Tensor pow(float exponent);

    Tensor mean();

    Tensor abs();

    Tensor crossEntropy(Tensor target);

    Tensor binaryCrossEntropy(Tensor target);

    Tensor div(Tensor other);

    Tensor add(float scalar);

    Tensor zerosLike();

    Tensor sqrt();

    Tensor cast(DType dtype);

    Tensor to(DeviceType device);

    default Tensor toFP32() {
        return cast(DType.F32);
    }

    float item();

    void backward();

    Tensor grad();

    void setGrad(Tensor grad);

    boolean requiresGrad();

    void setRequiresGrad(boolean requiresGrad);

    default void release() {
    }

    static Tensor randn(long... shape) {
        return TensorFactory.randn(shape);
    }

    static Tensor zeros(long... shape) {
        return TensorFactory.zeros(shape);
    }

    static Tensor ones(long... shape) {
        return TensorFactory.ones(shape);
    }

    static Tensor full(float value, long... shape) {
        return TensorFactory.full(value, shape);
    }

    static Tensor of(float[] data, long... shape) {
        return TensorFactory.of(data, shape);
    }

    // Common operations
    Tensor relu();
    Tensor sigmoid();
    Tensor tanh();
    Tensor log();
    Tensor exp();
    Tensor silu();
    Tensor flatten();
    Tensor unsqueeze(int dim);
    Tensor squeeze();
    Tensor transpose();
    Tensor transpose(int dim0, int dim1);

    long numel();
}
