package tech.kayys.gollek.compiler.neural;

import java.util.Comparator;
import java.util.List;

public final class NeuralCompilerV2 {
    public record KernelFeatures(String opId, long[] shape, String backend) {}
    public record KernelConfig(int tileSize, int vectorWidth) {}

    private final KernelModel model;

    public NeuralCompilerV2(KernelModel model) {
        this.model = model;
    }

    public KernelConfig select(KernelFeatures f, List<KernelConfig> configs) {
        return configs.stream()
            .min(Comparator.comparingDouble(cfg -> model.predict(f, cfg)))
            .orElseThrow();
    }
}
