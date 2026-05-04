package tech.kayys.gollek.train;

import tech.kayys.gollek.core.tensor.Tensor;

public final class PrecisionManager {
    private final PrecisionMode mode;

    public PrecisionManager(PrecisionMode mode) {
        this.mode = mode;
    }

    public Tensor castForward(Tensor t) {
        if (mode == PrecisionMode.FP32)
            return t;
        return t.cast(mode); // assume backend handles it
    }

    public Tensor castBackward(Tensor t) {
        // gradients often kept in FP32
        return t.toFP32();
    }
}