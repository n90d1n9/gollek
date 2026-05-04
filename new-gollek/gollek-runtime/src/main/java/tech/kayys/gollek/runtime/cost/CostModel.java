package tech.kayys.gollek.runtime.cost;

import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.runtime.Device;
import tech.kayys.gollek.runtime.kernel.*;

public interface CostModel {
    double estimate(
            GOp op,
            KernelCandidate kernel,
            Device device);
}