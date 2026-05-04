package tech.kayys.gollek.core.backend;

import tech.kayys.gollek.core.tensor.Tensor;

public interface ComputeBackend {
    Tensor add(Tensor a, Tensor b);

    Tensor mul(Tensor a, float scalar);

    Tensor matmul(Tensor a, Tensor b);

    Tensor reshape(Tensor a, long... newShape);

    Tensor attention(Tensor Q, Tensor K, Tensor V);

    Tensor sub(Tensor a, Tensor b);

    Tensor div(Tensor a, float scalar);

    Tensor softmax(Tensor a);
}