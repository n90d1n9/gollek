package tech.kayys.gollek.conversion;

public final class SafeTensorToIR implements ModelLoader {
@Override
public GollekGraph load(Path path) {
Map<String, Tensor> weights = loadWeights(path);
List<GollekNode> nodes = new ArrayList<>();
// example transformer block
nodes.add(new MatMulNode(...));
nodes.add(new AttentionNode(...));
nodes.add(new MatMulNode(...));
return new GollekGraph(nodes);
}
}
