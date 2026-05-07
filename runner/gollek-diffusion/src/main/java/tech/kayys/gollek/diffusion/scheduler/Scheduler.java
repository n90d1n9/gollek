package tech.kayys.gollek.diffusion.scheduler;
import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;


import tech.kayys.gollek.core.tensor.Tensor;

/**
 * SCHEDULER (FINAL, CORRECT DDIM)
 */
public interface Scheduler {
    Tensor step(Tensor x_t, Tensor eps, int tIndex);

    int[] timesteps();
}