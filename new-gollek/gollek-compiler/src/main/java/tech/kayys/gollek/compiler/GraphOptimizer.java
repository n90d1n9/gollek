package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.*;
import java.util.*;

public final class GraphOptimizer {
public GGraph optimize(GGraph graph) {
List<GOp> ops = new ArrayList<>(graph.ops());
ops = fuseElementwise(ops);
ops = fuseLinear(ops);
ops = fuseAttention(ops);
return new GGraph(
ops,
graph.outputs(),
graph.inputs()
);
}
private List<GOp> fuseElementwise(List<GOp> ops) {
// pattern match: add → gelu → mul
// replace with fused op
return ops;
}
private List<GOp> fuseLinear(List<GOp> ops) {
    return ops;
}
private List<GOp> fuseAttention(List<GOp> ops) {
return ops;
}
}