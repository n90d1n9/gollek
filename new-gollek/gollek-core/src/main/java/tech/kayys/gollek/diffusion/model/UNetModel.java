package tech.kayys.gollek.diffusion.model;

import tech.kayys.gollek.core.tensor.Tensor;

public interface UNetModel {
    Tensor predict(Tensor latents, Tensor embedding, int timestep);
}