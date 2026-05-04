package tech.kayys.gollek.core.data;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Unified data contract for all Gollek runtime data types.
 */
public sealed interface GData permits GTensor, GScalar, GHandle {
    String id();
}
