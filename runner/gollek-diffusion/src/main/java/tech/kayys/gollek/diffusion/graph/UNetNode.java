package tech.kayys.gollek.diffusion.graph;
import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.model.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;


import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.diffusion.model.UNetModel;

public final class UNetNode extends AbstractNode {
    private final Node latents;
    private final Node embedding;
    private final int timestep;
    private final UNetModel model;

    public UNetNode(Node latents, Node embedding, int timestep, UNetModel model) {
        this.latents = latents;
        this.embedding = embedding;
        this.timestep = timestep;
        this.model = model;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        return model.predict(
                latents.eval(ctx),
                embedding.eval(ctx),
                timestep);
    }
}
