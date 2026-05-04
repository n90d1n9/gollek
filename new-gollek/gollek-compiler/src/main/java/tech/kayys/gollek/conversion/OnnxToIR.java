package tech.kayys.gollek.conversion;

public final class OnnxToIR implements ModelLoader {
    @Override
    public GollekGraph load(Path path) {
        // parse ONNX graph
        // map ops → GollekNode
        return new GollekGraph(nodes);
    }
}