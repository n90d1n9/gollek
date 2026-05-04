package tech.kayys.gollek.compiler.neural;

import java.util.ArrayList;
import java.util.List;

public final class DatasetCollector {
    
    public record KernelTrainingSample(
        NeuralCompilerV2.KernelFeatures features,
        NeuralCompilerV2.KernelConfig config,
        double latency
    ) {}

    private final List<KernelTrainingSample> samples = new ArrayList<>();

    public void record(NeuralCompilerV2.KernelFeatures f, NeuralCompilerV2.KernelConfig c, double latency) {
        samples.add(new KernelTrainingSample(f, c, latency));
    }

    public List<KernelTrainingSample> dataset() {
        return List.copyOf(samples);
    }
    
    public void clear() {
        samples.clear();
    }
}
