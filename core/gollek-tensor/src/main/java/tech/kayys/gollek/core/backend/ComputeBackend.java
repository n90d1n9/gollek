package tech.kayys.gollek.core.backend;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.List;

/**
 * Interface for backend implementations to provide tensor operations.
 * This is located in gollek-tensor to avoid circular dependencies with Tensor.
 */
public interface ComputeBackend {
    Tensor add(Tensor a, Tensor b);

    Tensor sub(Tensor a, Tensor b);

    Tensor mul(Tensor a, float scalar);

    Tensor div(Tensor a, float scalar);

    Tensor matmul(Tensor a, Tensor b);

    Tensor reshape(Tensor a, long... newShape);

    Tensor slice(Tensor a, long[] offsets, long[] sizes);

    List<Tensor> split(Tensor a, int axis, int parts);

    Tensor attention(Tensor Q, Tensor K, Tensor V);

    Tensor softmax(Tensor a);

    Tensor pow(Tensor a, float exponent);

    Tensor mean(Tensor a);

    Tensor abs(Tensor a);

    Tensor crossEntropy(Tensor pred, Tensor target);

    Tensor binaryCrossEntropy(Tensor pred, Tensor target);

    Tensor cast(Tensor a, tech.kayys.gollek.core.tensor.DType dtype);

    Tensor to(Tensor a, tech.kayys.gollek.core.tensor.DeviceType device);

    Tensor mul(Tensor a, Tensor b);

    Tensor div(Tensor a, Tensor b);

    Tensor addScalar(Tensor a, float scalar);

    Tensor zerosLike(Tensor a);

    Tensor sqrt(Tensor a);
    Tensor relu(Tensor a);
    Tensor sigmoid(Tensor a);
    Tensor tanh(Tensor a);
    Tensor log(Tensor a);
    Tensor exp(Tensor a);
    Tensor silu(Tensor a);
    Tensor flatten(Tensor a);
    Tensor unsqueeze(Tensor a, int dim);
    Tensor squeeze(Tensor a);
    Tensor transpose(Tensor a);
    Tensor transpose(Tensor a, int dim0, int dim1);
    long numel(Tensor a);
}
