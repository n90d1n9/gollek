package tech.kayys.gollek.core.tensor;

import tech.kayys.gollek.core.tensor.Tensor;

public interface WeightAdapter {
    /**
     * Get weight tensor by name
     * 
     * @param name name of the tensor
     * @return Tensor
     * 
     */
    Tensor getWeight(String name);

    /**
     * Get number of layers
     * 
     * @return number of layers
     */
    int numLayers();

    /**
     * Get hidden size
     * 
     * @return hidden size
     */
    int hiddenSize();

    /**
     * Get number of attention heads
     * 
     * @return number of attention heads
     */
    int numHeads();
}
