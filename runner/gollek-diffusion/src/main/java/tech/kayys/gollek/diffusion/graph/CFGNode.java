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
import tech.kayys.gollek.core.graph.ExecutionContext;

/**
 * CLASSIFIER-FREE GUIDANCE (CFG)
 * 
 * @author bhangun
 */
public final class CFGNode extends AbstractNode {
    private final Node cond;
    private final Node uncond;
    private final float scale;

    public CFGNode(Node cond, Node uncond, float scale) {
        this.cond = cond;
        this.uncond = uncond;
        this.scale = scale;
    }

    @Override
    protected Tensor compute(ExecutionContext ctx) {
        Tensor uc = uncond.eval(ctx);
        Tensor c = cond.eval(ctx);
        return uc.add(
                c.sub(uc).mul(scale));
    }
}