package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.lang.foreign.Arena;

/**
 * Java-native GGUF backend using DirectForwardPass and AccelTensor.
 */
public class JavaNativeGgufBackend implements GgufBackend {
    
    private final GGUFModel model;
    private final Map<String, AccelTensor> weights;
    private final ModelConfig config;
    private final ModelArchitecture arch;
    
    public JavaNativeGgufBackend(Path modelPath) throws Exception {
        this.model = GGUFLoader.loadModel(modelPath);
        
        // TODO: Bridge GGUF metadata to ModelConfig
        this.config = new ModelConfig(); 
        this.arch = null; // TODO: Resolve arch
        
        Arena arena = Arena.ofAuto();
        List<TransformerLayerWeights> layers = GGUFLoader.loadLayers(model, arena);
        this.weights = GgufWeightAdapter.adaptWeights(model, layers, arch);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        // Implementation will call DirectForwardPass
        return RunnerResult.failed("Java-native execution logic pending final integration");
    }

    @Override
    public void close() {
        weights.values().forEach(AccelTensor::close);
        model.close();
    }
}
