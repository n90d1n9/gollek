package tech.kayys.gollek.nn;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.core.tensor.DeviceType;
import java.util.List;

public abstract class Module {
    public abstract Tensor forward(Tensor input);

    public void to(DeviceType device) {
        // Default implementation
    }

    public void zeroGrad() {
        // Default implementation
    }

    public List<Tensor> parameters() {
        return List.of();
    }
}
